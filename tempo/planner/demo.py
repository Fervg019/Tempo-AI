"""Compare strategies on many unseen factory scenarios (a first version of the final results table).

    python -m tempo.planner.demo                        # standard fleets
    python -m tempo.planner.demo --preset scarce        # few technicians, parts often late, more machines
    python -m tempo.planner.demo --preset healthy       # machines mostly far from failure
    python -m tempo.planner.demo --scenarios 100

Data is split so nothing is graded on data it learned from:
  forecaster trained on machines      seed 0
  calibration (range fix) machines    seed 20000
  alarm threshold tuned on fleets     seed 70000+   (a fair, tuned baseline, picked BEFORE testing)
  results measured on fleets          seed 50000+   (never seen by anything above)
"""
from __future__ import annotations

import argparse

import numpy as np

from tempo.forecast import CalibratedForecaster, QuantileForecaster, latest_rows
from tempo.forecast.baseline import QUANTILES
from tempo.simulator import generate_run_to_failure, make_scenario

from .plan import failures_under, plan_fleet, plan_from_samples, realized_cost
from .policies import alarm_threshold, run_to_failure

TEST_SEED = 50_000
TUNE_SEED = 70_000
ALARM_GRID = (7, 10, 14, 21, 30, 45, 60)
PRESETS = {
    "standard": dict(machines=20, tech_weekday=2, tech_weekend=1, parts_delay=0.4, age_low=0.35, age_high=0.95),
    "scarce": dict(machines=30, tech_weekday=1, tech_weekend=0, parts_delay=0.5, age_low=0.35, age_high=0.95),
    "healthy": dict(machines=20, tech_weekday=2, tech_weekend=1, parts_delay=0.4, age_low=0.05, age_high=0.8),
}


def build_models():
    base = QuantileForecaster().fit(generate_run_to_failure(300, seed=0))
    cal = CalibratedForecaster(base).fit(generate_run_to_failure(200, seed=20_000, first_id=200_000))
    return base, cal


def make_sc(seed: int, a) -> object:
    return make_scenario(seed, n_machines=a.machines, horizon=a.horizon, age_range=(a.age_low, a.age_high),
                         tech_weekday=a.tech_weekday, tech_weekend=a.tech_weekend, parts_delay_prob=a.parts_delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--scenarios", type=int, default=60)
    ap.add_argument("--tune-scenarios", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    for k, v in PRESETS["standard"].items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    print("training forecaster and calibrating ...")
    base, cal = build_models()
    med = list(QUANTILES).index(0.50)

    # 1) pick the alarm threshold on separate "practice" fleets
    tune_cost = {t: [] for t in ALARM_GRID}
    for k in range(args.tune_scenarios):
        sc = make_sc(TUNE_SEED + k, args)
        qc = cal.predict_from_features(latest_rows(sc.histories))
        for t in ALARM_GRID:
            tune_cost[t].append(realized_cost(sc, alarm_threshold(sc, qc[:, med], float(t))))
    best_t = min(ALARM_GRID, key=lambda t: np.mean(tune_cost[t]))

    # 2) measure everything on fleets nobody has seen
    names = ["run_to_failure", f"alarm<{best_t}d (tuned)", "tempo v1 (raw, no tails)",
             "tempo + tails", "tempo + calibration", "oracle (knows truth)"]
    res: dict[str, list[tuple[float, int, int]]] = {n: [] for n in names}
    for k in range(args.scenarios):
        sc = make_sc(TEST_SEED + k, args)
        feats = latest_rows(sc.histories)
        qb, qc = base.predict_from_features(feats), cal.predict_from_features(feats)
        H = sc.horizon
        plans = {
            names[0]: run_to_failure(sc),
            names[1]: alarm_threshold(sc, qc[:, med], float(best_t)),
            names[2]: plan_fleet(sc, base, qb, args.samples, seed=k, backend=args.backend, tails=False).service_day,
            names[3]: plan_fleet(sc, base, qb, args.samples, seed=k, backend=args.backend).service_day,
            names[4]: plan_fleet(sc, cal, qc, args.samples, seed=k, backend=args.backend).service_day,
            names[5]: plan_from_samples(sc.true_rul[:, None].astype(float), sc.demand, sc.technicians,
                                        sc.parts_ready_day, sc.costs).service_day,
        }
        for n, d in plans.items():
            res[n].append((realized_cost(sc, d), failures_under(sc, d), int((d < H).sum())))

    print(f"\npreset={args.preset}: {args.scenarios} unseen fleets of {args.machines} machines, "
          f"{args.horizon}-day horizon, technicians {args.tech_weekday}/weekday {args.tech_weekend}/weekend, "
          f"parts late for {args.parts_delay:.0%}")
    print(f"alarm threshold tuned on {args.tune_scenarios} separate fleets: {best_t} days\n")
    print(f"{'strategy':26s} {'avg cost':>11s} {'+/- 2 s.e.':>11s} {'failures':>9s} {'repairs':>8s}")
    for n, rows in res.items():
        c = np.array([r[0] for r in rows])
        se = 2 * c.std(ddof=1) / np.sqrt(len(c))
        print(f"{n:26s} {c.mean():11,.0f} {se:11,.0f} {np.mean([r[1] for r in rows]):9.2f} "
              f"{np.mean([r[2] for r in rows]):8.2f}")

    alarm = np.array([r[0] for r in res[names[1]]])
    for n in (names[2], names[3], names[4]):
        t = np.array([r[0] for r in res[n]])
        diff = alarm - t                       # positive = this Tempo variant is cheaper than the alarm
        se = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"{n:26s} vs tuned alarm: {100 * diff.mean() / alarm.mean():+6.1f}% cost "
              f"(+/- {100 * se / alarm.mean():.1f}), cheaper in {np.mean(diff > 0):.0%} of fleets")
    rtf = np.array([r[0] for r in res[names[0]]])
    t = np.array([r[0] for r in res[names[4]]])
    print(f"\nTempo (calibrated) vs run-to-failure: {100 * (rtf.mean() - t.mean()) / rtf.mean():.1f}% lower cost")


if __name__ == "__main__":
    main()
