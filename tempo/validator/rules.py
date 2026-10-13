"""The rulebook: plain, non-AI checks that every plan must pass before a human sees it.

Why this exists
---------------
An LLM can write a convincing maintenance plan that quietly breaks a rule (books a repair
on a day nobody is on shift, uses a part that has not arrived, skips lockout/tagout, or
states a number that is not in the data). The LLM PROPOSES, this module DECIDES.

Three checks
------------
1. validate_plan         the schedule itself: technicians, spare parts, limits, risk
2. validate_work_order   one machine's work order (usually drafted by the LLM): same day as the
                         plan, and the mandatory safety steps are all present
3. check_numbers         every number in the LLM's explanation must exist in the real data

Findings are either "error" (the plan must not go to a human as-is) or "warning" (a human
should look, but the plan is allowed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from tempo.simulator import Scenario

ERROR = "error"
WARNING = "warning"

# Every work order must cover each item. A step counts if it contains any of the keywords.
SAFETY_CHECKLIST: dict[str, tuple[str, ...]] = {
    "lockout/tagout": ("lockout", "tagout", "lock out", "loto"),
    "energy isolation": ("isolate", "de-energ", "deenerg", "zero energy", "disconnect power"),
    "protective equipment": ("ppe", "protective equipment", "safety glasses", "gloves"),
    "test before restart": ("test run", "function test", "restart check", "verify operation", "test before"),
}


@dataclass(frozen=True)
class Policy:
    """Company rules on top of the physical limits. All numbers are assumptions to be set by the user."""
    max_machines_per_day: int | None = None     # extra cap on repairs per day (None = technicians only)
    risk_threshold: float = 0.20                # chance of failing before its service that needs a human
    safety_checklist: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(SAFETY_CHECKLIST))


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str                  # "error" or "warning"
    message: str
    machine_id: int | None = None
    day: int | None = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing blocks the plan (warnings are allowed)."""
        return not self.errors

    def extend(self, other: "Report") -> "Report":
        self.findings.extend(other.findings)
        return self

    def rules_failed(self) -> set[str]:
        return {f.rule for f in self.errors}

    def to_dicts(self) -> list[dict]:
        return [f.__dict__.copy() for f in self.findings]

    def summary(self) -> str:
        if not self.findings:
            return "All checks passed."
        lines = [f"{'ERROR' if f.severity == ERROR else 'warning'} [{f.rule}] {f.message}" for f in self.findings]
        return "\n".join(lines)


def _add(rep: Report, rule: str, severity: str, msg: str, machine_id=None, day=None) -> None:
    rep.findings.append(Finding(rule, severity, msg, None if machine_id is None else int(machine_id),
                                None if day is None else int(day)))


# ---------------------------------------------------------------------------
# 1. The schedule
# ---------------------------------------------------------------------------

def validate_plan(scenario: Scenario, service_day: Sequence[int], p_fail_before_service: Sequence[float] | None = None,
                  policy: Policy | None = None) -> Report:
    """service_day[i] = day to repair machine i; == horizon means 'no repair planned'."""
    policy = policy or Policy()
    rep = Report()
    H = scenario.horizon
    n = scenario.n_machines
    ids = scenario.machine_ids
    day = np.asarray(service_day)

    if day.shape != (n,):
        _add(rep, "SHAPE", ERROR, f"plan has {day.size} entries but there are {n} machines")
        return rep
    if not np.issubdtype(day.dtype, np.integer):
        if not np.all(np.equal(np.mod(day, 1), 0)):
            _add(rep, "SHAPE", ERROR, "service days must be whole numbers")
            return rep
        day = day.astype(int)

    # R1: inside the planning window
    for i in np.where((day < 0) | (day > H))[0]:
        _add(rep, "RANGE", ERROR, f"machine {ids[i]}: day {day[i]} is outside 0..{H} (use {H} for 'no repair')",
             ids[i], day[i])
    if rep.errors:
        return rep

    # R2: technicians on shift
    for d in range(H):
        booked = np.where(day == d)[0]
        if len(booked) > scenario.technicians[d]:
            who = ", ".join(str(ids[i]) for i in booked)
            why = ("nobody is on shift" if scenario.technicians[d] == 0
                   else f"only {scenario.technicians[d]} technician(s) on shift")
            _add(rep, "TECHNICIANS", ERROR, f"day {d}: {len(booked)} repair(s) booked (machines {who}) but {why}", day=d)

    # R3: spare part has arrived
    for i in np.where(day < H)[0]:
        if day[i] < scenario.parts_ready_day[i]:
            _add(rep, "PARTS", ERROR,
                 f"machine {ids[i]}: repair on day {day[i]} but its spare part arrives on day {scenario.parts_ready_day[i]}",
                 ids[i], day[i])

    # R4: company cap on repairs per day
    if policy.max_machines_per_day is not None:
        for d in range(H):
            c = int((day == d).sum())
            if c > policy.max_machines_per_day:
                _add(rep, "DAILY_CAP", ERROR,
                     f"day {d}: {c} repairs exceed the company limit of {policy.max_machines_per_day} per day", day=d)

    # R5: risky machines that are not fixed in time need a human decision (warning, not an error:
    #     the validator cannot create technicians or parts)
    if p_fail_before_service is not None:
        p = np.asarray(p_fail_before_service, dtype=float)
        for i in np.where(p >= policy.risk_threshold)[0]:
            if day[i] >= H:
                _add(rep, "RISK", WARNING,
                     f"machine {ids[i]}: no repair planned and {p[i]:.0%} chance it fails within {H} days",
                     ids[i])
            else:
                _add(rep, "RISK", WARNING,
                     f"machine {ids[i]}: {p[i]:.0%} chance it fails before its repair on day {day[i]}", ids[i], day[i])
    return rep


# ---------------------------------------------------------------------------
# 2. A work order (usually drafted by the LLM)
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("machine_id", "service_day", "steps", "safety_steps")


def validate_work_order(order: dict, scenario: Scenario, service_day: Sequence[int],
                        policy: Policy | None = None) -> Report:
    """Checks one drafted work order against the real plan and the safety checklist."""
    policy = policy or Policy()
    rep = Report()
    H = scenario.horizon
    if not isinstance(order, dict):
        _add(rep, "ORDER_FORMAT", ERROR, "work order is not a JSON object")
        return rep
    missing = [k for k in _REQUIRED_KEYS if k not in order]
    if missing:
        _add(rep, "ORDER_FORMAT", ERROR, f"work order is missing: {', '.join(missing)}")
        return rep

    mid = order["machine_id"]
    ids = list(map(int, scenario.machine_ids))
    try:
        mid_int = int(mid)
    except (TypeError, ValueError):
        _add(rep, "ORDER_FORMAT", ERROR, f"machine_id {mid!r} is not a number")
        return rep
    if mid_int not in ids:
        _add(rep, "ORDER_MACHINE", ERROR, f"machine {mid_int} does not exist in this fleet", mid_int)
        return rep
    idx = ids.index(mid_int)
    planned = int(np.asarray(service_day)[idx])

    if planned >= H:
        _add(rep, "ORDER_UNPLANNED", ERROR, f"machine {mid_int} has no repair in the plan, so it cannot have a work order",
             mid_int)
    else:
        try:
            day = int(order["service_day"])
        except (TypeError, ValueError):
            day = None
        if day != planned:
            _add(rep, "ORDER_DAY", ERROR,
                 f"machine {mid_int}: work order says day {order['service_day']!r} but the plan says day {planned}",
                 mid_int, planned)

    steps = order["steps"]
    safety = order["safety_steps"]
    errors_so_far = len(rep.errors)
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
        _add(rep, "ORDER_FORMAT", ERROR, "'steps' must be a non-empty list of text instructions", mid_int)
    if not isinstance(safety, list) or not all(isinstance(s, str) for s in safety):
        _add(rep, "ORDER_FORMAT", ERROR, "'safety_steps' must be a list of text instructions", mid_int)
    if len(rep.errors) > errors_so_far:      # the steps are unreadable, so the safety check cannot run
        return rep

    text = " ".join(safety + steps).lower()
    for item, keywords in policy.safety_checklist.items():
        if not any(k in text for k in keywords):
            _add(rep, "SAFETY", ERROR, f"machine {mid_int}: mandatory safety step missing: {item}", mid_int)
    return rep


# ---------------------------------------------------------------------------
# 3. Numbers in the LLM's explanation
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    """Every number in the text as (original text, value, is_percent). Commas and $ are stripped."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        is_pct = raw.endswith("%")
        try:
            out.append((raw, float(raw.strip("$%").replace(",", "")), is_pct))
        except ValueError:
            continue
    return out


def facts_from_plan(scenario: Scenario, service_day: Sequence[int], p_fail_before_service: Sequence[float] | None = None,
                    expected_cost: float | None = None, expected_cost_by_machine: Sequence[float] | None = None,
                    extra: Iterable[float] = ()) -> list[float]:
    """The numbers an explanation is allowed to mention (days, machine numbers, counts, costs, chances)."""
    H = scenario.horizon
    day = np.asarray(service_day)
    facts: list[float] = [H, len(day), int((day < H).sum()), int((day >= H).sum())]
    facts += [float(i) for i in scenario.machine_ids]
    facts += [float(d) for d in day]
    facts += [float(d) for d in np.unique(scenario.technicians)]
    facts += [float(d) for d in np.unique(scenario.parts_ready_day)]
    if p_fail_before_service is not None:
        facts += [100.0 * float(p) for p in p_fail_before_service]          # as percentages
    if expected_cost is not None:
        facts.append(float(expected_cost))
    if expected_cost_by_machine is not None:
        facts += [float(c) for c in expected_cost_by_machine]
    facts += [float(x) for x in extra]
    return facts


def check_numbers(text: str, allowed: Iterable[float], tolerance: float = 0.01) -> Report:
    """Warn about any number in `text` that is not in `allowed` (within ~1%, or 0.5 for small numbers).

    Percentages are compared as percentages (allowed values for chances are 0-100).
    Costs may be written in thousands ("$42k") only if the exact figure is listed in `allowed` too.
    """
    rep = Report()
    allowed = [float(a) for a in allowed]
    for raw, value, is_pct in extract_numbers(text):
        ok = any(abs(value - a) <= max(0.5, tolerance * abs(a)) for a in allowed)
        if not ok:
            _add(rep, "UNVERIFIED_NUMBER", WARNING,
                 f"the text says {raw!r}, which does not match any number in the plan data")
    return rep
