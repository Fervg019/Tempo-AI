"""The logic behind the app, with no web code in it (so it can be tested directly).

Engine          builds a factory scenario from a few settings, forecasts, plans, validates
ExplainService  asks the LLM for the explanation and work orders as a background job, with a cache on
                disk, daily and per-visitor limits, and a no-AI fallback so the screen never breaks
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tempo.forecast import latest_rows
from tempo.llm import LLMError, build_facts, explain_plan, template_work_order
from tempo.planner import failures_under, plan_fleet, plan_from_samples, realized_cost, run_to_failure
from tempo.simulator import make_scenario
from tempo.validator import validate_plan

PROMPT_VERSION = "v1"
FLEETS = {"stressed": (0.35, 0.95), "healthy": (0.05, 0.8)}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]     # make_scenario: day 0 is a Monday

PRESETS = {
    "short_staffed": dict(seed=3, machines=12, horizon=21, tech_weekday=1, tech_weekend=0, parts_delay=0.4,
                          fleet="stressed"),
    "well_staffed": dict(seed=3, machines=12, horizon=21, tech_weekday=2, tech_weekend=1, parts_delay=0.4,
                         fleet="stressed"),
    "healthy_fleet": dict(seed=5, machines=12, horizon=21, tech_weekday=2, tech_weekend=1, parts_delay=0.4,
                          fleet="healthy"),
}


class BadRequest(ValueError):
    pass


@dataclass(frozen=True)
class Params:
    seed: int = 3
    machines: int = 12
    horizon: int = 21
    tech_weekday: int = 1
    tech_weekend: int = 0
    parts_delay: float = 0.4
    fleet: str = "stressed"

    @classmethod
    def parse(cls, raw: dict) -> "Params":
        def integer(name, lo, hi, default):
            v = raw.get(name, default)
            try:
                if isinstance(v, bool) or (isinstance(v, float) and v != int(v)):
                    raise ValueError
                v = int(v)
            except (TypeError, ValueError):
                raise BadRequest(f"{name} must be a whole number") from None
            if not lo <= v <= hi:
                raise BadRequest(f"{name} must be between {lo} and {hi}")
            return v

        d = cls()
        try:
            delay = float(raw.get("parts_delay", d.parts_delay))
        except (TypeError, ValueError):
            raise BadRequest("parts_delay must be a number") from None
        if not 0.0 <= delay <= 0.9:
            raise BadRequest("parts_delay must be between 0 and 0.9")
        fleet = raw.get("fleet", d.fleet)
        if fleet not in FLEETS:
            raise BadRequest(f"fleet must be one of {sorted(FLEETS)}")
        return cls(seed=integer("seed", 0, 99999, d.seed), machines=integer("machines", 4, 40, d.machines),
                   horizon=integer("horizon", 14, 30, d.horizon),
                   tech_weekday=integer("tech_weekday", 0, 4, d.tech_weekday),
                   tech_weekend=integer("tech_weekend", 0, 3, d.tech_weekend),
                   parts_delay=round(delay, 2), fleet=fleet)

    def as_dict(self) -> dict:
        return asdict(self)


class Computed:
    """Everything derived from one set of settings."""

    def __init__(self, p: Params, models):
        self.p = p
        self.sc = make_scenario(60_000 + p.seed, n_machines=p.machines, horizon=p.horizon,
                                age_range=FLEETS[p.fleet], tech_weekday=p.tech_weekday,
                                tech_weekend=p.tech_weekend, parts_delay_prob=p.parts_delay)
        self.feats = latest_rows(self.sc.histories)
        self.q = models.predict_from_features(self.feats)
        self.plan = plan_fleet(self.sc, models, self.q, n_samples=3000, seed=p.seed)
        self.facts = build_facts(self.sc, self.plan, self.q, self.feats)
        self.report = validate_plan(self.sc, self.plan.service_day, self.plan.p_fail_before_service)

    def plan_rows(self) -> list[dict]:
        return [{"machine_id": int(m["machine_id"]), "service_day": m["service_day"]} for m in self.facts["machines"]]

    def plan_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.plan_rows(), sort_keys=True).encode()).hexdigest()


def _findings(report) -> list[dict]:
    return [{"severity": f.severity, "rule": f.rule, "message": f.message, "machine_id": f.machine_id,
             "day": f.day} for f in report.findings]


class Engine:
    def __init__(self, models, cache_size: int = 64):
        self.models = models
        self._cache: "OrderedDict[Params, Computed]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._heavy = threading.BoundedSemaphore(3)      # at most 3 plans being computed at once

    def compute(self, p: Params) -> Computed:
        with self._lock:
            if p in self._cache:
                self._cache.move_to_end(p)
                return self._cache[p]
        with self._heavy:
            with self._lock:
                if p in self._cache:                     # someone else just finished the same plan
                    return self._cache[p]
            c = Computed(p, self.models)
        with self._lock:
            self._cache[p] = c
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return c

    def plan_view(self, p: Params) -> dict:
        c = self.compute(p)
        sc, plan, H = c.sc, c.plan, c.sc.horizon
        machines = []
        for i, m in enumerate(c.facts["machines"]):
            machines.append({
                **m,
                "parts_ready_day": int(sc.parts_ready_day[i]),
                "cost_if_no_repair": int(round(float(plan.cost_matrix[i, H]))),
            })
        by_day: dict[int, list[int]] = {}
        for m in machines:
            if m["service_day"] is not None:
                by_day.setdefault(m["service_day"], []).append(m["machine_id"])
        days = [{"day": d, "label": WEEKDAYS[d % 7], "technicians": int(sc.technicians[d]),
                 "demand": round(float(sc.demand[d]), 2), "repairs": by_day.get(d, [])} for d in range(H)]
        overview = dict(c.facts["overview"])
        overview["expected_cost_do_nothing"] = int(round(float(plan.cost_matrix[:, H].sum())))
        return {"params": p.as_dict(), "overview": overview, "machines": machines, "days": days,
                "validator": {"ok": c.report.ok, "findings": _findings(c.report)},
                "plan_sha256": c.plan_hash(), "planner": {"samples": plan.n_samples, "backend": plan.backend,
                                                          "seconds": round(plan.seconds_total, 3)}}

    def reveal(self, p: Params) -> dict:
        """Simulation only: score the plan against the hidden truth the planner never saw."""
        c = self.compute(p)
        sc = c.sc
        oracle = plan_from_samples(sc.true_rul[:, None].astype(float), sc.demand, sc.technicians,
                                   sc.parts_ready_day, sc.costs).service_day

        def score(days):
            return {"cost": int(round(realized_cost(sc, days))), "failures": failures_under(sc, days),
                    "repairs": int((days < sc.horizon).sum())}
        return {"tempo": score(c.plan.service_day), "do_nothing": score(run_to_failure(sc)),
                "oracle": score(oracle),
                "true_days_left": {int(i): int(round(float(t))) for i, t in zip(sc.machine_ids, sc.true_rul)},
                "note": "Simulation only. The planner never sees these true failure times; the oracle is a "
                        "cheat that does, shown as the best anyone could do."}


# ---------------------------------------------------------------------------
# AI explanation as a background job
# ---------------------------------------------------------------------------

class OfflineLLM:
    """Stand-in used when no key is set or limits are reached: every call fails, so the pipeline's own
    fallback (plain summary + template work orders) produces the output."""
    calls = 0

    def chat_json(self, *a, **k):
        raise LLMError("offline")


class ExplainService:
    def __init__(self, engine: Engine, llm, cache_dir, *, model_name: str = "none", daily_cap: int = 40,
                 per_visitor_hour: int = 6, max_orders: int = 15, clock=time.time):
        self.engine, self.llm, self.model_name = engine, llm, model_name
        self.cache_dir = Path(cache_dir)
        self.daily_cap, self.per_visitor_hour, self.max_orders = daily_cap, per_visitor_hour, max_orders
        self._clock = clock
        self._jobs: dict[str, dict] = {}
        self._starts: list[float] = []
        self._visitor: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ---- keys and cache ----
    def key(self, p: Params) -> str:
        blob = json.dumps({**p.as_dict(), "model": self.model_name, "prompt": PROMPT_VERSION}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def _disk_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_disk(self, key: str) -> dict | None:
        try:
            return json.loads(self._disk_path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_disk(self, key: str, result: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._disk_path(key).write_text(json.dumps(result, indent=1), encoding="utf-8")
        except OSError:
            pass

    # ---- public ----
    def status(self, p: Params) -> dict:
        key = self.key(p)
        with self._lock:
            job = self._jobs.get(key)
        if job:
            return dict(job, key=key)
        cached = self._load_disk(key)
        if cached:
            return {"status": "done", "key": key, "result": dict(cached, from_cache=True)}
        return {"status": "none", "key": key}

    def start(self, p: Params, visitor: str = "local") -> dict:
        key = self.key(p)
        st = self.status(p)
        if st["status"] in ("running", "done"):
            return st
        reason = self._why_offline(visitor)
        if reason:
            result = self._generate(p, OfflineLLM(), mode="fallback", notice=reason)
            job = {"status": "done", "result": result}
            with self._lock:
                self._jobs[key] = job
            return dict(job, key=key)
        with self._lock:
            now = self._clock()
            self._starts.append(now)
            self._visitor.setdefault(visitor, []).append(now)
            self._jobs[key] = {"status": "running", "started": now}
        threading.Thread(target=self._run_ai, args=(p, key), daemon=True).start()
        return {"status": "running", "key": key}

    def _why_offline(self, visitor: str) -> str | None:
        if self.llm is None:
            return "No AI key is configured on this server, so the plain no-AI version is shown."
        now = self._clock()
        with self._lock:
            self._starts = [t for t in self._starts if now - t < 86400]
            hits = [t for t in self._visitor.get(visitor, []) if now - t < 3600]
            self._visitor[visitor] = hits
            if len(self._starts) >= self.daily_cap:
                return "The demo's daily AI limit is used up, so the plain no-AI version is shown."
            if len(hits) >= self.per_visitor_hour:
                return "You reached the hourly AI limit for this demo, so the plain no-AI version is shown."
        return None

    def _run_ai(self, p: Params, key: str) -> None:
        try:
            result = self._generate(p, self.llm, mode="ai")
        except Exception as e:  # noqa: BLE001 - never let a background thread die silently
            result = self._generate(p, OfflineLLM(), mode="fallback",
                                    notice=f"The AI step failed ({type(e).__name__}); the plain no-AI version is shown.")
        if result["mode"] == "ai":
            self._save_disk(key, result)
        with self._lock:
            self._jobs[key] = {"status": "done", "result": result}

    # ---- the actual work ----
    def _generate(self, p: Params, llm, mode: str, notice: str | None = None) -> dict:
        c = self.engine.compute(p)
        r = explain_plan(llm, c.sc, c.plan, c.q, c.feats, max_orders=self.max_orders)
        orders = {int(k): dict(v) for k, v in r.work_orders.items()}
        sources = {int(k): v for k, v in r.order_source.items()}
        for m in c.facts["machines"]:                       # beyond the AI order cap: plain template orders
            mid = int(m["machine_id"])
            if m["service_day"] is not None and mid not in orders:
                orders[mid] = template_work_order(mid, m["service_day"], m["main_warning_sensor"])
                sources[mid] = "template"
        rows = []
        for mid in sorted(orders, key=lambda k: (orders[k]["service_day"], k)):
            o = orders[mid]
            rows.append({"machine_id": mid, "service_day": o["service_day"], "title": o.get("title"),
                         "steps": o["steps"], "safety_steps": o["safety_steps"], "source": sources[mid]})
        ai_orders = sum(1 for s in sources.values() if s != "template")
        if mode == "ai" and r.llm_calls > 0 and ai_orders == 0 and r.llm_errors:
            mode, notice = "fallback", f"The AI did not respond usefully ({r.llm_errors[0][:120]}); showing the plain version."
        if mode == "ai" and r.llm_errors:
            notice = "Some AI replies were unusable and were replaced by the rulebook's fallbacks."
        return {"mode": mode, "notice": notice, "model": self.model_name if mode == "ai" else None,
                "summary": r.summary, "reasons": {str(k): v for k, v in r.reasons.items()},
                "work_orders": rows,
                "unverified_numbers": [f.message for f in r.explanation_report.warnings],
                "stats": r.stats, "llm_calls": r.llm_calls, "llm_errors": r.llm_errors[:6],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plan_sha256": c.plan_hash()}
