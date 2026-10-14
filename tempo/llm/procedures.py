"""Example repair procedures, looked up by which sensor is drifting the most.

IMPORTANT: these are short EXAMPLE procedures written for this demo. They are not real
manufacturer instructions and must not be used on real equipment. In a real plant this library
would be the plant's own manuals and maintenance records.

The lookup is deliberately simple and deterministic (the sensor that has moved furthest toward
its worn-out level picks the procedure), so the LLM never has to guess which procedure applies.
"""
from __future__ import annotations

from dataclasses import dataclass

GENERAL_SAFETY = [
    "Apply lockout/tagout to the machine's main power isolator and keep the key with the technician.",
    "Isolate all stored energy (electrical, pressure, spring) and verify zero energy before opening any guard.",
    "Wear protective equipment: safety glasses, gloves and hearing protection.",
]
GENERAL_CLOSEOUT = [
    "Refit all guards, remove locks and tags, and run a function test before handing the machine back to production.",
    "Record what was replaced and the readings before and after in the maintenance log.",
]


@dataclass(frozen=True)
class Procedure:
    key: str
    title: str
    steps: tuple[str, ...]


PROCEDURES: dict[str, Procedure] = {
    "vibration": Procedure("bearing", "Bearing inspection and replacement", (
        "Remove the guard and inspect the bearing housings for heat marks, play and noise.",
        "Replace worn bearings and check shaft alignment.",
        "Re-grease to the machine's specified grease type.",
    )),
    "temperature": Procedure("cooling", "Cooling and lubrication service", (
        "Inspect fans, cooling fins and coolant lines for blockage or leaks.",
        "Clean or replace filters and top up coolant.",
        "Check lubricant level and condition.",
    )),
    "pressure": Procedure("seals", "Seal and pressure line inspection", (
        "Inspect pressure lines, fittings and seals for leaks or cracking.",
        "Replace worn seals and damaged hose sections.",
        "Check the pressure relief valve operates freely.",
    )),
    "current": Procedure("motor", "Motor and drive check", (
        "Inspect motor terminals and cables for heat damage or loose connections.",
        "Check drive settings and the motor's insulation condition.",
        "Check the load for jams or binding that would raise current draw.",
    )),
    "noise_db": Procedure("gear", "Gear and coupling inspection", (
        "Inspect gears, belts and couplings for wear, looseness or misalignment.",
        "Replace worn drive components and tighten fixings.",
        "Check lubrication of all moving contacts.",
    )),
    "oil_particles": Procedure("lubricant", "Lubricant change and wear-particle check", (
        "Drain the lubricant and inspect the drained oil and filter for metal particles.",
        "Flush the system, replace the filter and refill with the specified lubricant.",
        "Note the particle findings so an early-stage bearing or gear fault can be followed up.",
    )),
}


def procedure_for(sensor: str) -> Procedure:
    return PROCEDURES.get(sensor, PROCEDURES["vibration"])


def template_work_order(machine_id: int, service_day: int, sensor: str) -> dict:
    """A work order built from the procedure alone, with no AI. Always passes the validator's checks.

    Used as the fallback when the LLM's draft cannot be fixed, so a human never sees an invalid order.
    """
    proc = procedure_for(sensor)
    return {
        "machine_id": int(machine_id),
        "service_day": int(service_day),
        "title": proc.title,
        "steps": list(proc.steps) + list(GENERAL_CLOSEOUT),
        "safety_steps": list(GENERAL_SAFETY),
    }
