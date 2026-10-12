"""Train and evaluate the baseline forecaster on simulated machines.

Run from the project folder:
    python -m tempo.forecast.train
    python -m tempo.forecast.train --n-train 400 --n-test 100 --seed 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from tempo.simulator import generate_run_to_failure

from .baseline import QuantileForecaster, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models/baseline.joblib"))
    args = ap.parse_args()

    # Different seeds so the test machines are truly unseen.
    train = generate_run_to_failure(args.n_train, seed=args.seed)
    test = generate_run_to_failure(args.n_test, seed=args.seed + 10_000, first_id=100_000)
    print(f"train rows: {len(train):,}  test rows: {len(test):,}")

    t0 = time.time()
    model = QuantileForecaster().fit(train)
    print(f"trained in {time.time() - t0:.1f}s")

    m = evaluate(model, test)
    print("\n                      all rows    last 30 days before failure")
    print(f"error (days, median)  {m['all_mae_days']:8.2f}    {m['late_mae_days']:8.2f}")
    print(f"'guess the average'   {m['naive_mae_days']:8.2f}")
    print(f"90% range holds truth {m['all_coverage_90']:8.1%}    {m['late_coverage_90']:8.1%}   (ideal ~90%)")
    print(f"50% range holds truth {m['all_coverage_50']:8.1%}    {m['late_coverage_50']:8.1%}   (ideal ~50%)")
    print(f"avg 90% range width   {m['all_width_90_days']:8.1f}    {m['late_width_90_days']:8.1f}  days")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nsaved model to {args.out}")


if __name__ == "__main__":
    main()
