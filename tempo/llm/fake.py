"""A stand-in for the real model, so everything can be tested and demoed with no key and no internet.

mode="good"    answers correctly
mode="sloppy"  makes the mistakes real LLMs make: changes a repair day, forgets lockout/tagout,
               invents a cost. Used to show (and test) that the validator catches them, and that a
               second attempt or the template fixes them.
"""
from __future__ import annotations

import json

from .procedures import GENERAL_CLOSEOUT, GENERAL_SAFETY, procedure_for


class FakeLLM:
    def __init__(self, mode: str = "good"):
        assert mode in ("good", "sloppy")
        self.mode = mode
        self.calls = 0

    def chat_json(self, messages, max_tokens: int = 1800, temperature: float = 0.2) -> dict:
        self.calls += 1
        system = messages[0]["content"]
        payload = json.loads(messages[-1]["content"])
        return self._orders(payload) if "work orders" in system else self._explain(payload)

    # ---- explanation ----
    def _explain(self, facts: dict) -> dict:
        ov = facts["overview"]
        summary = (f"{ov['machines_with_repair_planned']} of {ov['machines_total']} machines need a repair in the next "
                   f"{ov['planning_days']} days. The expected total cost is ${ov['expected_total_cost']:,}.")
        if self.mode == "sloppy":
            summary += " This plan saves roughly $987,654 compared with doing nothing."
        machines = []
        for m in facts["machines"]:
            if m["service_day"] is None:
                continue
            machines.append({"machine_id": m["machine_id"], "reason": (
                f"Repair on day {m['service_day']}: its {m['main_warning_sensor']} reading is "
                f"{m['sensor_wear_percent']}% of the way to its worn-out level, with a "
                f"{m['chance_of_failing_before_repair_percent']}% chance of failing before then.")})
        return {"summary": summary, "machines": machines}

    # ---- work orders ----
    def _orders(self, payload: dict) -> dict:
        fixing = bool(payload.get("fix_these_problems"))
        orders = []
        for k, m in enumerate(payload["machines"]):
            proc = procedure_for(m["main_warning_sensor"])
            order = {"machine_id": m["machine_id"], "service_day": m["service_day"],
                     "steps": list(proc.steps) + list(GENERAL_CLOSEOUT), "safety_steps": list(GENERAL_SAFETY)}
            if self.mode == "sloppy" and not fixing:
                if k % 3 == 0:
                    order["service_day"] = m["service_day"] + 1                       # changes the planned day
                elif k % 3 == 1:
                    order["safety_steps"] = ["Wear protective equipment (safety glasses and gloves)."]
                    order["steps"] = list(proc.steps)                                 # no lockout, no test run
            orders.append(order)
        return {"work_orders": orders}
