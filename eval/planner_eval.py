"""How sensitive is Tempo's advantage to the money assumptions?

    python -m eval.planner_eval                       # everything (about 15-25 minutes)
    python -m eval.planner_eval --quick               # tiny run to check the script works

Every cost number in the simulator is an assumption, so this answers the obvious question: "what if
your costs are wrong?" For each fleet type (standard / scarce / healthy) and each cost setting we compare,
on fleets nothing has seen before (same split as tempo.planner.demo):

    run to failure    never service
    tuned alarm       service when the median forecast drops below N days; N is tuned on practice
                      fleets under the TRUE costs (so the baseline gets a small head start)
    Tempo             calibrated forecaster + Monte Carlo + exact scheduler
    oracle            knows the true failure days (unreachable, shown as the floor)

Two ways of running Tempo:
    knows costs       the planner is told the true cost numbers
    assumes base      the planner uses the default numbers while reality uses the setting being tested
                      (what happens if the company's real costs differ from what we assumed)

Results are written to eval/results/planner_sensitivity.json and .md
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from tempo.forecast import latest_rows
from tempo.forecast.baseline import QUANTILES
from tempo.planner import alarm_threshold, failures_under, plan_fleet, plan_from_samples, realized_cost, run_to_failure
from tempo.planner.demo import ALARM_GRID, PRESETS, TEST_SEED, TUNE_SEED, build_models
from tempo.simulator import CostModel, make_scenario

BASE = CostModel()
COSTS = {
    "base": BASE,
    "cheap_failures": dataclasses.replace(BASE, unplanned_downtime_per_day=6_000.0, unplanned_repair=4_000.0),
    "costly_failures": dataclasses.replace(BASE, unplanned_downtime_per_day=40_000.0, unplanned_repair=16_000.0),
    "costly_planned_work": dataclasses.replace(BASE, planned_downtime_per_day=8_000.0, planned_repair=5_000.0,
                                               wasted_life_per_day=400.0),
}
COST_LABEL = {
    "base": "default costs",
    "cheap_failures": "failures cheaper",
    "costly_failures": "failures much costlier",
    "costly_planned_work": "planned repairs costlier",
}


def failure_ratio(c: CostModel) -> float:
    """How many times worse a surprise failure is than a planned repair (ignoring demand effects)."""
    return (c.unplanned_downtime_per_day * c.unplanned_downtime_days + c.unplanned_repair) / \
           (c.planned_downtime_per_day * c.service_days + c.planned_repair)


def fleet(seed: int, p: dict, horizon: int, costs: CostModel):
    return make_scenario(seed, n_machines=p["machines"], horizon=horizon, age_range=(p["age_low"], p["age_high"]),
                         tech_weekday=p["tech_weekday"], tech_weekend=p["tech_weekend"],
                         parts_delay_prob=p["parts_delay"], costs=costs)


def run_cell(cal, preset: str, cost_name: str, assumed_name: str, n_test: int, n_tune: int, samples: int,
             horizon: int) -> dict:
    p, true_costs, assumed = PRESETS[preset], COSTS[cost_name], COSTS[assumed_name]
    med = list(QUANTILES).index(0.50)

    tune = {t: [] for t in ALARM_GRID}
    for k in range(n_tune):
        sc = fleet(TUNE_SEED + k, p, horizon, true_costs)
        q = cal.predict_from_features(latest_rows(sc.histories))
        for t in ALARM_GRID:
            tune[t].append(realized_cost(sc, alarm_threshold(sc, q[:, med], float(t))))
    best_t = min(ALARM_GRID, key=lambda t: np.mean(tune[t]))

    rows = {"run_to_failure": [], "alarm": [], "tempo": [], "oracle": []}
    for k in range(n_test):
        sc = fleet(TEST_SEED + k, p, horizon, true_costs)
        feats = latest_rows(sc.histories)
        q = cal.predict_from_features(feats)
        planning_view = dataclasses.replace(sc, costs=assumed)          # what the planner believes costs are
        days = {
            "run_to_failure": run_to_failure(sc),
            "alarm": alarm_threshold(sc, q[:, med], float(best_t)),
            "tempo": plan_fleet(planning_view, cal, q, samples, seed=k).service_day,
            "oracle": plan_from_samples(sc.true_rul[:, None].astype(float), sc.demand, sc.technicians,
                                        sc.parts_ready_day, true_costs).service_day,
        }
        for name, d in days.items():
            rows[name].append((realized_cost(sc, d), failures_under(sc, d)))

    cost = {n: np.array([r[0] for r in v]) for n, v in rows.items()}
    diff = cost["alarm"] - cost["tempo"]                                  # positive = Tempo cheaper
    se = 2 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return {
        "preset": preset, "cost_setting": cost_name, "planner_assumed": assumed_name,
        "failure_to_planned_cost_ratio": round(failure_ratio(true_costs), 1),
        "fleets": n_test, "alarm_threshold_days": best_t,
        "avg_cost": {n: round(float(c.mean())) for n, c in cost.items()},
        "avg_failures": {n: round(float(np.mean([r[1] for r in v])), 2) for n, v in rows.items()},
        "tempo_vs_alarm_pct": round(100 * float(diff.mean()) / float(cost["alarm"].mean()), 1),
        "tempo_vs_alarm_pct_pm2se": round(100 * float(se) / float(cost["alarm"].mean()), 1),
        "tempo_cheaper_share": round(float(np.mean(diff > 0)), 2),
        "tempo_vs_run_to_failure_pct": round(100 * float((cost["run_to_failure"] - cost["tempo"]).mean()
                                                         / cost["run_to_failure"].mean()), 1),
    }


def to_markdown(results: list[dict]) -> str:
    out = ["| Fleet | Costs | Failure vs planned repair | Planner believes | Run to failure | Tuned alarm | **Tempo** | Oracle | Tempo vs alarm | Cheaper in |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r["avg_cost"]
        sign = "cheaper" if r["tempo_vs_alarm_pct"] >= 0 else "dearer"
        out.append(f"| {r['preset']} | {COST_LABEL[r['cost_setting']]} | {r['failure_to_planned_cost_ratio']}x | "
                   f"{'true costs' if r['planner_assumed'] == r['cost_setting'] else 'default costs (wrong)'} | "
                   f"{a['run_to_failure']:,} | {a['alarm']:,} | **{a['tempo']:,}** | {a['oracle']:,} | "
                   f"{abs(r['tempo_vs_alarm_pct']):.1f}% {sign} (+/- {r['tempo_vs_alarm_pct_pm2se']:.1f}) | "
                   f"{r['tempo_cheaper_share']:.0%} of fleets |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=40)
    ap.add_argument("--tune-scenarios", type=int, default=25)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--presets", nargs="*", default=list(PRESETS))
    ap.add_argument("--costs", nargs="*", default=list(COSTS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    if a.quick:
        a.scenarios, a.tune_scenarios, a.samples = 3, 3, 500

    print("training forecaster and calibrating ...", flush=True)
    _, cal = build_models()
    results, t0 = [], time.time()
    jobs = [(pr, cn, an) for pr in a.presets for cn in a.costs for an in ([cn] if cn == "base" else [cn, "base"])]
    for i, (pr, cn, an) in enumerate(jobs, 1):
        r = run_cell(cal, pr, cn, an, a.scenarios, a.tune_scenarios, a.samples, a.horizon)
        results.append(r)
        print(f"[{i}/{len(jobs)}] {pr:9s} {cn:20s} planner uses {an:20s} Tempo vs alarm {r['tempo_vs_alarm_pct']:+6.1f}% "
              f"(+/- {r['tempo_vs_alarm_pct_pm2se']:.1f})   {time.time() - t0:5.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "planner_sensitivity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "planner_sensitivity.md").write_text(to_markdown(results) + "\n", encoding="utf-8")
    print("\n" + to_markdown(results))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
