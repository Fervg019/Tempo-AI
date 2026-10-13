"""Simple strategies Tempo is compared against. All return a service day per machine
(== horizon means 'no service')."""
from __future__ import annotations

import numpy as np

from tempo.simulator import Scenario


def run_to_failure(scenario: Scenario) -> np.ndarray:
    """Never service. Fix only after it breaks."""
    return np.full(scenario.n_machines, scenario.horizon, dtype=int)


def alarm_threshold(scenario: Scenario, median_rul: np.ndarray, threshold_days: float = 14.0) -> np.ndarray:
    """Simple alarm: when a machine's median forecast drops below a line, service it as soon as
    a technician and its part are available. Most urgent machines go first."""
    H = scenario.horizon
    day = np.full(scenario.n_machines, H, dtype=int)
    capacity = scenario.technicians.astype(int).copy()
    for i in np.argsort(median_rul):
        if median_rul[i] >= threshold_days:
            break
        for d in range(int(scenario.parts_ready_day[i]), H):
            if capacity[d] > 0:
                capacity[d] -= 1
                day[i] = d
                break
    return day
