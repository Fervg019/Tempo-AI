"""Try the explainer end to end.

    python -m tempo.llm.demo --check             test your Token Factory key (lists models, one hello call)
    python -m tempo.llm.demo --fake sloppy       offline: a pretend LLM that makes typical mistakes
    python -m tempo.llm.demo --fake good         offline: a pretend LLM that behaves
    python -m tempo.llm.demo                     the real model (needs RADEON_API_KEY in .env)

It builds a small fleet, plans the repairs, then asks the LLM to explain the plan and draft work
orders. Every draft goes through the validator, and the stats at the end show what it caught.
"""
from __future__ import annotations

import argparse
import sys
import time

from tempo.forecast import latest_rows
from tempo.planner import plan_fleet
from tempo.planner.demo import build_models
from tempo.simulator import make_scenario

from .client import LLMClient, LLMError
from .explain import explain_plan
from .fake import FakeLLM


def check_connection() -> int:
    try:
        client = LLMClient.from_env()
        print(client)
        models = client.models()
        print(f"{len(models)} models available: {', '.join(models)}")
        if client.model not in models:
            print(f"WARNING: model {client.model!r} is not in the list. Set TEMPO_MODEL in .env to one of the above.")
        t0 = time.time()
        reply = client.chat([{"role": "system", "content": "You are concise."},
                             {"role": "user", "content": "Say hello in one sentence."}], max_tokens=200)
        print(f"reply in {time.time() - t0:.1f}s: {reply.strip()}")
        print("OK: the key works.")
        return 0
    except LLMError as e:
        print(f"FAILED: {e}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only test the connection")
    ap.add_argument("--fake", choices=["good", "sloppy"], help="use a pretend LLM (no key, no internet)")
    ap.add_argument("--machines", type=int, default=12)
    ap.add_argument("--max-orders", type=int, default=8, help="work orders to draft (keeps API calls low)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.check:
        return check_connection()
    try:
        llm = FakeLLM(args.fake) if args.fake else LLMClient.from_env()
    except LLMError as e:
        print(f"{e}\n(or use --fake good / --fake sloppy to try it offline)")
        return 1

    print("training forecaster and planning repairs (about 30 seconds) ...")
    base, cal = build_models()
    sc = make_scenario(60_000 + args.seed, n_machines=args.machines, horizon=21, tech_weekday=1, tech_weekend=0)
    feats = latest_rows(sc.histories)
    q = cal.predict_from_features(feats)
    plan = plan_fleet(sc, cal, q, n_samples=3000, seed=args.seed)
    print(f"plan: {int((plan.service_day < sc.horizon).sum())} of {sc.n_machines} machines get a repair, "
          f"expected cost {plan.expected_cost:,.0f}\n")

    try:
        result = explain_plan(llm, sc, plan, q, feats, max_orders=args.max_orders)
    except LLMError as e:
        print(f"FAILED: {e}")
        return 1

    print("SUMMARY\n  " + result.summary + "\n")
    print("REASONS (first 4)")
    for mid, text in list(result.reasons.items())[:4]:
        print(f"  machine {mid}: {text}")
    print("\nWORK ORDERS")
    for mid, order in result.work_orders.items():
        print(f"  machine {mid:>2}  day {order['service_day']:>2}  source={result.order_source[mid]:<12} "
              f"{len(order['steps'])} steps, {len(order['safety_steps'])} safety steps")
    print("\nWHAT THE RULEBOOK DID")
    for k, v in result.stats.items():
        print(f"  {k}: {v:.0%}" if k.endswith("rate") else f"  {k}: {v}")
    print(f"  llm calls: {result.llm_calls}")
    for f in result.explanation_report.warnings:
        print(f"  flagged: {f.message}")
    print("  plan check:", "passed" if result.plan_report.ok else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
