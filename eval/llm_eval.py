"""How often does the AI model break the rulebook, and does the rulebook catch it?

    python -m eval.llm_eval                      # real model (needs RADEON_API_KEY in .env), about 8-15 minutes
    python -m eval.llm_eval --fake sloppy        # offline check that the script works (numbers are NOT results)
    python -m eval.llm_eval --fleets 3           # smaller

For each of several fleets we plan the repairs and ask the model to draft work orders twice:
    rules told      the prompt spells out the safety checklist (how the app normally runs)
    rules hidden    the prompt does NOT mention the checklist, so the validator, not the prompt, is what
                    would catch a missing lockout/tagout. This is a stress test of the safety net with a
                    deliberately naive prompt, not a general "how sloppy is the model" rate.
Every draft goes through the validator. We count what it caught and what it could not fix.

What this does NOT measure: whether the explanation's reasoning is true. The validator checks numbers and
safety items, not the logic of a sentence. Results are written to eval/results/llm_eval.json and .md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tempo.forecast import latest_rows
from tempo.llm import FakeLLM, LLMClient, LLMError, explain_plan
from tempo.planner import plan_fleet
from tempo.planner.demo import build_models
from tempo.simulator import make_scenario

SETUPS = [  # (machines, weekday techs, weekend techs, parts delay, age range)
    (12, 1, 0, 0.4, (0.35, 0.95)), (12, 2, 1, 0.4, (0.35, 0.95)), (12, 2, 1, 0.4, (0.05, 0.8)),
    (16, 1, 0, 0.5, (0.35, 0.95)), (10, 2, 1, 0.3, (0.35, 0.95)), (14, 1, 1, 0.4, (0.2, 0.9)),
]


def classify(message: str) -> str:
    m = message.lower()
    if "mandatory safety step missing" in m:
        item = m.split("missing:", 1)[1].split("(", 1)[0].strip()
        return f"missing {item}"
    if "work order says day" in m:
        return "wrong service day"
    if "is missing:" in m or "not a json" in m or "must be" in m:
        return "malformed work order"
    return "other"


def run(llm, cal, fleets: int, max_orders: int, tell: bool) -> dict:
    totals = dict(orders=0, valid_first=0, valid_repair=0, template=0, unverified=0, calls=0, api_errors=0)
    kinds: dict[str, int] = {}
    still: dict[str, int] = {}
    errors: list[str] = []
    for k in range(fleets):
        machines, wd, we, delay, age = SETUPS[k % len(SETUPS)]
        sc = make_scenario(80_000 + k, n_machines=machines, horizon=21, age_range=age, tech_weekday=wd,
                           tech_weekend=we, parts_delay_prob=delay)
        feats = latest_rows(sc.histories)
        q = cal.predict_from_features(feats)
        plan = plan_fleet(sc, cal, q, n_samples=3000, seed=k)
        try:
            r = explain_plan(llm, sc, plan, q, feats, max_orders=max_orders, tell_model_the_rules=tell)
        except LLMError as e:
            print(f"   fleet {k}: skipped ({e})", flush=True)
            continue
        s = r.stats
        totals["orders"] += int(s["work_orders"])
        totals["valid_first"] += int(s["llm_valid_first_try"])
        totals["valid_repair"] += int(s["llm_valid_after_repair"])
        totals["template"] += int(s["template_fallback"])
        totals["unverified"] += int(s["unverified_numbers_in_text"])
        totals["calls"] += r.llm_calls
        totals["api_errors"] += len(r.llm_errors)
        for msgs in r.first_try_problems.values():
            for msg in msgs:
                kinds[classify(msg)] = kinds.get(classify(msg), 0) + 1
        for msgs in r.repair_problems.values():
            for msg in msgs:
                still[classify(msg)] = still.get(classify(msg), 0) + 1
        errors += [f"fleet {k}: {e}" for e in r.llm_errors]
        print(f"   fleet {k}: {int(s['llm_valid_first_try'])}/{int(s['work_orders'])} valid first try, "
              f"{int(s['template_fallback'])} template, {r.llm_calls} calls"
              + (f", {len(r.llm_errors)} API/format problems" if r.llm_errors else ""), flush=True)
    n = max(1, totals["orders"])
    return {"rules_told_in_prompt": tell, **totals, "problem_kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
            "still_wrong_after_repair": dict(sorted(still.items(), key=lambda kv: -kv[1])), "api_problems": errors[:20],
            "first_try_valid_rate": round(totals["valid_first"] / n, 3),
            "valid_after_repair_rate": round((totals["valid_first"] + totals["valid_repair"]) / n, 3),
            "template_rate": round(totals["template"] / n, 3)}


def to_markdown(model: str, rows: list[dict]) -> str:
    out = [f"Model: {model}. Same fleets and plans in both rows.", "",
           "| Prompt | Work orders | Valid as written | Valid after one repair try | Replaced by template | Numbers the checker flagged | What it caught |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        kinds = ", ".join(f"{k} ({v})" for k, v in r["problem_kinds"].items()) or "nothing"
        out.append(f"| {'safety rules spelled out' if r['rules_told_in_prompt'] else 'safety rules hidden'} | {r['orders']} | "
                   f"{r['first_try_valid_rate']:.0%} | {r['valid_after_repair_rate']:.0%} | {r['template_rate']:.0%} | "
                   f"{r['unverified']} | {kinds} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", type=int, default=6)
    ap.add_argument("--only", choices=["told", "hidden"], help="run just one of the two prompts")
    ap.add_argument("--max-orders", type=int, default=8)
    ap.add_argument("--fake", choices=["good", "sloppy"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    try:
        llm = FakeLLM(a.fake) if a.fake else LLMClient.from_env()
    except LLMError as e:
        raise SystemExit(f"{e}\n(or try --fake sloppy to test the script offline)")
    model = f"offline stand-in ({a.fake}), NOT a result" if a.fake else getattr(llm, "model", "unknown")
    print("training forecaster ...", flush=True)
    _, cal = build_models()
    rows, t0 = [], time.time()
    for tell in {None: (True, False), 'told': (True,), 'hidden': (False,)}[a.only]:
        print(f"\nsafety rules {'spelled out' if tell else 'hidden'} in the prompt:", flush=True)
        rows.append(run(llm, cal, a.fleets, a.max_orders, tell))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = ("llm_eval_FAKE" if a.fake else "llm_eval") + (f"_{a.only}_only" if a.only else "")
    (out / f"{stem}.json").write_text(json.dumps({"model": model, "results": rows}, indent=1), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(model, rows) + "\n", encoding="utf-8")
    print("\n" + to_markdown(model, rows))
    print(f"\n{time.time() - t0:.0f}s, saved to {out}")


if __name__ == "__main__":
    main()
