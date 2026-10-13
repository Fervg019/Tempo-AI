"""Compare strategies on many unseen factory scenarios (a first, rough version of the final results table).

    python -m tempo.planner.demo
    python -m tempo.planner.demo --scenarios 100 --machines 20
    python -m tempo.planner.demo --age-low 0.05 --age-high 0.8     # a healthier fleet

The simple alarm is shown with several thresholds, and the best one is used as the fair
baseline. (The best threshold is picked on the same scenarios, which flatters the baseline,
so the comparison is conservative for Tempo.)
"""
from __future__ import annotations

import argparse

import numpy as np

from tempo.forecast import QuantileForecaster, latest_rows
from tempo.forecast.baseline import QUANTILES
from tempo.simulator import generate_run_to_failure, make_scenario

from .plan import failures_under, plan_fleet, realized_cost
from .policies import alarm_threshold, run_to_failure

# Scenario seeds start far from the training seeds so test machines are never training machines.
SCENARIO_SEED_OFFSET = 50_000
ALARM_THRESHOLDS = (7, 14, 21, 30, 45)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=50)
    ap.add_argument("--machines", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--age-low", type=float, default=0.35)
    ap.add_argument("--age-high", type=float, default=0.95)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    args = ap.parse_args()

    print("training forecaster ...")
    model = QuantileForecaster().fit(generate_run_to_failure(300, seed=0))
    med_col = list(QUANTILES).index(0.50)

    names = ["run_to_failure"] + [f"alarm<{t}d" for t in ALARM_THRESHOLDS] + ["tempo"]
    results: dict[str, list[tuple[float, int, int]]] = {n: [] for n in names}
    for k in range(args.scenarios):
        sc = make_scenario(SCENARIO_SEED_OFFSET + k, n_machines=args.machines, horizon=args.horizon,
                           age_range=(args.age_low, args.age_high))
        q = model.predict_from_features(latest_rows(sc.histories))
        H = sc.horizon

        plans = {"run_to_failure": run_to_failure(sc)}
        for t in ALARM_THRESHOLDS:
            plans[f"alarm<{t}d"] = alarm_threshold(sc, q[:, med_col], threshold_days=float(t))
        plans["tempo"] = plan_fleet(sc, model, q, n_samples=args.samples, seed=k, backend=args.backend).service_day
        for name, days in plans.items():
            results[name].append((realized_cost(sc, days), failures_under(sc, days), int((days < H).sum())))

    mean_cost = {n: float(np.mean([r[0] for r in rows])) for n, rows in results.items()}
    best_alarm = min((n for n in names if n.startswith("alarm")), key=mean_cost.get)

    print(f"\n{args.scenarios} unseen scenarios, {args.machines} machines each, {args.horizon}-day horizon, "
          f"fleet age range {args.age_low}-{args.age_high}")
    print(f"{'strategy':16s} {'avg cost':>12s} {'+/- (2 s.e.)':>13s} {'failures':>9s} {'repairs':>8s}")
    for name, rows in results.items():
        cost = np.array([r[0] for r in rows])
        se = 2 * cost.std(ddof=1) / np.sqrt(len(cost)) if len(cost) > 1 else 0.0
        tag = "  <- best alarm" if name == best_alarm else ""
        print(f"{name:16s} {cost.mean():12,.0f} {se:13,.0f} {np.mean([r[1] for r in rows]):9.2f} "
              f"{np.mean([r[2] for r in rows]):8.2f}{tag}")
    t, base, a = mean_cost["tempo"], mean_cost["run_to_failure"], mean_cost[best_alarm]
    print(f"\nTempo vs run-to-failure:        {100 * (base - t) / base:5.1f}% lower cost")
    print(f"Tempo vs best simple alarm:     {100 * (a - t) / a:5.1f}% lower cost")


if __name__ == "__main__":
    main()
