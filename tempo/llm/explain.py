"""Turn a computed repair plan into plain language and work orders, with the rulebook in charge.

Flow
----
 1. build_facts      collect every real number about the plan (the ONLY numbers the LLM may use)
 2. explanation      one LLM call: a short summary + one reason per machine
                     -> numbers in the text are checked against the facts (unverified ones are flagged)
 3. work orders      LLM calls in small batches, one draft per machine from its example procedure
                     -> each draft goes through the validator (right day? safety checklist complete?)
                     -> a failed draft gets ONE repair attempt with the exact errors
                     -> still failing? a no-AI template order is used instead
 4. stats            how often the raw LLM draft was valid on the first try (a number for the pitch)

The LLM proposes. The validator decides. A human approves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from tempo.forecast.baseline import QUANTILES
from tempo.simulator import SENSOR_NAMES, Scenario
from tempo.simulator.factory import _GAIN
from tempo.validator import Report, check_numbers, facts_from_plan, validate_plan, validate_work_order

from .client import LLMError
from .procedures import GENERAL_SAFETY, procedure_for, template_work_order

EXPLAIN_SYSTEM = (
    "You are a maintenance planning assistant writing for a factory supervisor. An optimizer has ALREADY "
    "computed a repair plan. Your job is only to explain it in plain, short sentences. Rules: use ONLY numbers "
    "that appear in the facts; never invent machines, days, costs or chances; never change the plan; write "
    "costs as full numbers like $12,345 (never abbreviate). Day 0 is today. Repairs cannot happen on the days "
    "listed in days_with_no_technicians. Do NOT guess why the optimizer chose a day beyond what the facts show, "
    "and do NOT say a machine has a 0% chance or no risk unless its chance_of_failing_before_repair_percent is 0. "
    "Reply with ONLY a JSON object: "
    '{"summary": "3 to 5 sentences for the whole fleet", "machines": [{"machine_id": <id>, "reason": "at most 2 '
    'sentences"}]} with one entry per machine that has a repair planned, using the real machine_id values from the facts '
    '(<id> is a placeholder, not a real value).'
)
ORDER_SYSTEM = (
    "You draft maintenance work orders. For each machine you are given, base the steps on the procedure text "
    "provided and do not add tools, parts or numbers that are not in it. Rules: service_day must equal the "
    "given service_day exactly; safety_steps must cover lockout/tagout, isolating all energy, and protective "
    "equipment; steps must end with a function test before restart. Reply with ONLY a JSON object: "
    '{"work_orders": [{"machine_id": <id>, "service_day": <day>, "steps": ["..."], "safety_steps": ["..."]}]} '
    "with exactly one entry for EACH machine in the input, copying its machine_id and service_day from the input "
    "(<id> and <day> are placeholders, not real values)."
)
# Used to measure the model's RAW mistake rate: the same job, but the safety rules are NOT spelled out,
# so the validator (not the prompt) is what catches a missing lockout/tagout.
ORDER_SYSTEM_BARE = (
    "You draft maintenance work orders. For each machine you are given, base the steps on the procedure text "
    "provided and do not add tools, parts or numbers that are not in it. service_day must equal the given "
    "service_day exactly. Reply with ONLY a JSON object: "
    '{"work_orders": [{"machine_id": <id>, "service_day": <day>, "steps": ["..."], "safety_steps": ["..."]}]} '
    "with exactly one entry for EACH machine in the input, copying its machine_id and service_day from the input "
    "(<id> and <day> are placeholders, not real values)."
)


@dataclass
class ExplainedPlan:
    summary: str
    reasons: dict[int, str]
    work_orders: dict[int, dict]
    order_source: dict[int, str]              # "llm" | "llm_repaired" | "template"
    plan_report: Report
    explanation_report: Report
    stats: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    first_try_problems: dict[int, list[str]] = field(default_factory=dict)   # what the validator caught, per machine
    repair_problems: dict[int, list[str]] = field(default_factory=dict)      # what was still wrong after the repair try
    llm_errors: list[str] = field(default_factory=list)                      # API/parse errors that were absorbed


# ---------------------------------------------------------------------------
# 1. Facts
# ---------------------------------------------------------------------------

def _numbers_in(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers_in(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _numbers_in(v)]
    return []


def dominant_sensors(latest_features) -> list[tuple[str, int]]:
    """For each machine (rows of latest_rows()): the sensor furthest toward its worn-out level, and how far (%)."""
    out = []
    for _, row in latest_features.iterrows():
        frac = np.array([max(0.0, row[f"{s}_delta0"]) / g for s, g in zip(SENSOR_NAMES, _GAIN)])
        j = int(np.argmax(frac))
        out.append((SENSOR_NAMES[j], int(round(100 * min(1.0, frac[j])))))
    return out


def build_facts(scenario: Scenario, plan, quantile_values: np.ndarray, latest_features) -> dict:
    """Everything the LLM may talk about. `plan` is a tempo.planner.Plan."""
    H = scenario.horizon
    qs = list(QUANTILES)
    lo, mid, hi = (quantile_values[:, qs.index(q)] for q in (0.05, 0.50, 0.95))
    sensors = dominant_sensors(latest_features)
    machines = []
    for i, mid_id in enumerate(scenario.machine_ids):
        d = int(plan.service_day[i])
        sensor, wear = sensors[i]
        entry = {
            "machine_id": int(mid_id),
            "service_day": d if d < H else None,
            "chance_of_failing_before_repair_percent": int(round(100 * plan.p_fail_before_service[i])),
            "days_left_forecast": {"low": int(round(lo[i])), "likely": int(round(mid[i])), "high": int(round(hi[i]))},
            "main_warning_sensor": sensor,
            "sensor_wear_percent": wear,
            "expected_cost": int(round(plan.expected_cost_by_machine[i])),
        }
        if d < H:
            entry["technicians_on_that_day"] = int(scenario.technicians[d])
            entry["production_that_day"] = ("busy" if scenario.demand[d] >= 1.5
                                            else "quiet" if scenario.demand[d] < 0.8 else "normal")
        machines.append(entry)
    overview = {
        "planning_days": H,
        "machines_total": scenario.n_machines,
        "machines_with_repair_planned": int((plan.service_day < H).sum()),
        "machines_without_repair_planned": int((plan.service_day >= H).sum()),
        "expected_total_cost": int(round(plan.expected_cost)),
        "days_with_no_technicians": [int(d) for d in range(H) if int(scenario.technicians[d]) == 0],
    }
    return {"overview": overview, "machines": machines}


def allowed_numbers(scenario: Scenario, plan, facts: dict) -> list[float]:
    return facts_from_plan(scenario, plan.service_day, plan.p_fail_before_service, plan.expected_cost,
                           plan.expected_cost_by_machine, extra=_numbers_in(facts))


# ---------------------------------------------------------------------------
# 2-3. LLM steps
# ---------------------------------------------------------------------------

def _ask(llm, system: str, payload: dict, max_tokens: int) -> dict:
    return llm.chat_json([{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(payload, indent=1)}], max_tokens=max_tokens)


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def explain_plan(llm, scenario: Scenario, plan, quantile_values: np.ndarray, latest_features,
                 batch_size: int = 3, max_orders: int | None = None,
                 tell_model_the_rules: bool = True) -> ExplainedPlan:
    """Full pipeline. `llm` needs .chat_json(messages, max_tokens=...) (LLMClient or FakeLLM).

    tell_model_the_rules=True  (normal use) the prompt spells out the safety checklist.
    tell_model_the_rules=False (for measuring) it does not, so "first_try_valid_rate" is the model's
                               raw rate of breaking the rulebook, which is the number worth reporting.
    """
    H = scenario.horizon
    calls_before = getattr(llm, "calls", 0)
    facts = build_facts(scenario, plan, quantile_values, latest_features)
    plan_report = validate_plan(scenario, plan.service_day, plan.p_fail_before_service)
    allowed = allowed_numbers(scenario, plan, facts)
    by_id = {m["machine_id"]: m for m in facts["machines"]}

    # ---- explanation ----
    llm_errors: list[str] = []
    reasons: dict[int, str] = {}
    summary = ""
    try:
        reply = _ask(llm, EXPLAIN_SYSTEM, facts, max_tokens=1800)
        s = reply.get("summary")
        summary = s.strip() if isinstance(s, str) else ""
        items = reply.get("machines")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                mid = int(item["machine_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if mid in by_id and isinstance(item.get("reason"), str):
                reasons[mid] = item["reason"].strip()
    except LLMError as e:
        llm_errors.append(f"explanation call: {e}")
    ov = facts["overview"]
    if not summary:
        summary = (f"{ov['machines_with_repair_planned']} of {ov['machines_total']} machines have a repair planned "
                   f"within {ov['planning_days']} days. Expected total cost is ${ov['expected_total_cost']:,}.")
    for mid, m in by_id.items():
        if mid not in reasons and m["service_day"] is not None:
            reasons[mid] = (f"Repair on day {m['service_day']}: {m['main_warning_sensor']} is "
                            f"{m['sensor_wear_percent']}% of the way to its worn-out level.")
    explanation_report = check_numbers(" ".join([summary] + list(reasons.values())), allowed)

    # ---- work orders ----
    scheduled = [m for m in facts["machines"] if m["service_day"] is not None]
    scheduled.sort(key=lambda m: (m["service_day"], -m["chance_of_failing_before_repair_percent"]))
    if max_orders is not None:
        scheduled = scheduled[:max_orders]
    sday = plan.service_day
    ids = [int(x) for x in scenario.machine_ids]

    def request_orders(batch: list[dict], problems: dict[int, list[str]] | None = None) -> dict[int, dict]:
        entries = []
        for m in batch:
            e = {"machine_id": m["machine_id"], "service_day": m["service_day"],
                 "main_warning_sensor": m["main_warning_sensor"],
                 "procedure_title": procedure_for(m["main_warning_sensor"]).title,
                 "procedure_steps": list(procedure_for(m["main_warning_sensor"]).steps)}
            if tell_model_the_rules:
                e["required_safety_steps"] = GENERAL_SAFETY
            entries.append(e)
        payload = {"machines": entries}
        if problems:
            payload["fix_these_problems"] = {str(k): v for k, v in problems.items()}
        try:
            reply = _ask(llm, ORDER_SYSTEM if tell_model_the_rules else ORDER_SYSTEM_BARE, payload, max_tokens=2500)
        except LLMError as e:
            llm_errors.append(f"work-order call for machines {[m['machine_id'] for m in batch]}: {e}")
            return {}
        out = {}
        raw_orders = reply.get("work_orders")
        if not isinstance(raw_orders, list):
            llm_errors.append(f"work-order reply for machines {[m['machine_id'] for m in batch]} had no "
                              f"'work_orders' list (keys: {sorted(map(str, reply))[:6]})")
        for o in raw_orders if isinstance(raw_orders, list) else []:
            try:
                out[int(o["machine_id"])] = o
            except (KeyError, TypeError, ValueError):
                continue
        wanted = [m["machine_id"] for m in batch]
        if isinstance(raw_orders, list) and any(w not in out for w in wanted):
            first = raw_orders[0] if raw_orders else None
            shape = (sorted(map(str, first))[:8] if isinstance(first, dict) else type(first).__name__)
            llm_errors.append(f"asked for machines {wanted}; reply had {len(raw_orders)} order(s) matching "
                              f"{sorted(out)}; first item looks like: {shape}")
        return out

    drafts: dict[int, dict] = {}
    for batch in _chunks(scheduled, batch_size):
        drafts.update(request_orders(batch))

    orders: dict[int, dict] = {}
    source: dict[int, str] = {}
    failed: list[dict] = []
    problems: dict[int, list[str]] = {}
    first_try_valid = 0
    for m in scheduled:
        mid = m["machine_id"]
        rep = validate_work_order(drafts.get(mid, {}), scenario, sday)
        if rep.ok:
            orders[mid], source[mid] = drafts[mid], "llm"
            first_try_valid += 1
        else:
            failed.append(m)
            problems[mid] = [f.message for f in rep.errors]

    repaired = 0
    still_wrong: dict[int, list[str]] = {}
    for batch in _chunks(failed, batch_size):
        retry = request_orders(batch, {m["machine_id"]: problems[m["machine_id"]] for m in batch})
        for m in batch:
            mid = m["machine_id"]
            if mid not in retry:
                still_wrong[mid] = ["the model returned no work order for this machine in the repair reply"]
                continue
            rep2 = validate_work_order(retry[mid], scenario, sday)
            if rep2.ok:
                orders[mid], source[mid] = retry[mid], "llm_repaired"
                repaired += 1
            else:
                still_wrong[mid] = [f.message for f in rep2.errors]
    for m in failed:
        mid = m["machine_id"]
        if mid not in orders:
            orders[mid] = template_work_order(mid, m["service_day"], m["main_warning_sensor"])
            source[mid] = "template"

    n = max(1, len(scheduled))
    stats = {
        "work_orders": len(scheduled),
        "llm_valid_first_try": first_try_valid,
        "llm_valid_after_repair": repaired,
        "template_fallback": sum(1 for v in source.values() if v == "template"),
        "first_try_valid_rate": first_try_valid / n,
        "unverified_numbers_in_text": len(explanation_report.warnings),
        "problems_caught": sum(len(v) for v in problems.values()),
        "rules_in_prompt": 1.0 if tell_model_the_rules else 0.0,
    }
    return ExplainedPlan(summary, reasons, orders, source, plan_report, explanation_report, stats,
                         getattr(llm, "calls", 0) - calls_before, problems, still_wrong, llm_errors)
