"""Pick the cheapest repair schedule.

Key insight: given the cost matrix C[i, d], machines only interact through
technician capacity (only so many repairs can happen per day) and spare parts
(a repair cannot happen before its part arrives). That makes the schedule an
ASSIGNMENT problem, which the classic Hungarian algorithm solves exactly and
quickly. So the plan is optimal for the model, not just "good". The GPU does the
heavy part (building C); the assignment step is small.

"Optimal" always means: optimal under our forecast ranges and our cost
assumptions. It is not a promise about the real world.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tempo.simulator import CostModel, Scenario

from .cost import cost_matrix_numpy, cost_matrix_torch

_FORBIDDEN = 1e12


@dataclass
class Plan:
    service_day: np.ndarray               # (n,) day to service each machine; == horizon means "no service planned"
    horizon: int
    expected_cost: float                  # average cost over all simulated futures
    expected_cost_by_machine: np.ndarray  # (n,)
    p_fail_before_service: np.ndarray     # (n,) chance it fails before its planned service (or within horizon if none)
    cost_matrix: np.ndarray               # (n, H+1)
    n_samples: int
    seconds_cost_matrix: float
    seconds_total: float
    backend: str

    def scheduled(self) -> np.ndarray:
        return self.service_day < self.horizon


def solve_assignment(C: np.ndarray, technicians: np.ndarray, parts_ready_day: np.ndarray) -> np.ndarray:
    """Cheapest way to give every machine one day (or 'none'), respecting technicians and parts."""
    n, hp1 = C.shape
    H = hp1 - 1
    # One column per technician slot per day, plus n 'no service' columns that are always allowed.
    col_day = np.concatenate([np.full(int(technicians[d]), d) for d in range(H)] + [np.full(n, H)])
    cost = C[:, col_day].copy()
    too_early = (col_day[None, :] < parts_ready_day[:, None]) & (col_day[None, :] < H)
    cost[too_early] = _FORBIDDEN
    rows, cols = linear_sum_assignment(cost)
    service_day = np.full(n, H, dtype=int)
    service_day[rows] = col_day[cols]
    return service_day


def plan_from_samples(samples: np.ndarray, demand: np.ndarray, technicians: np.ndarray,
                      parts_ready_day: np.ndarray, costs: CostModel,
                      backend: str = "numpy", device: str = "auto") -> Plan:
    """samples: (n, S) possible failure times (days from now) for every machine."""
    t0 = time.perf_counter()
    if backend == "numpy":
        C = cost_matrix_numpy(samples, demand, costs)
    elif backend == "torch":
        C = cost_matrix_torch(samples, demand, costs, device=device)
    else:
        raise ValueError("backend must be 'numpy' or 'torch'")
    t1 = time.perf_counter()

    H = len(demand)
    n = samples.shape[0]
    day = solve_assignment(C, technicians, parts_ready_day)
    by_machine = C[np.arange(n), day]
    p_fail = np.array([(samples[i] < min(day[i], H)).mean() if day[i] < H else (samples[i] < H).mean()
                       for i in range(n)])
    return Plan(service_day=day, horizon=H, expected_cost=float(by_machine.sum()),
                expected_cost_by_machine=by_machine, p_fail_before_service=p_fail, cost_matrix=C,
                n_samples=samples.shape[1], seconds_cost_matrix=t1 - t0,
                seconds_total=time.perf_counter() - t0, backend=f"{backend}:{device}" if backend == "torch" else backend)


def plan_fleet(scenario: Scenario, forecaster, quantile_values: np.ndarray, n_samples: int = 5000,
               seed: int = 0, backend: str = "numpy", device: str = "auto", tails: bool = True) -> Plan:
    """Forecast ranges -> sampled failure times -> cheapest schedule for a scenario.

    tails=False reproduces the first version, which ignored the lowest/highest 5% of futures.
    """
    rng = np.random.default_rng(seed)
    samples = forecaster.sample_rul(quantile_values, n_samples, rng, tails=tails)
    return plan_from_samples(samples, scenario.demand, scenario.technicians, scenario.parts_ready_day,
                             scenario.costs, backend=backend, device=device)


def realized_cost(scenario: Scenario, service_day: np.ndarray) -> float:
    """What a schedule REALLY costs, using the machines' true (hidden) failure times.

    Only for scoring. The planner never sees true_rul.
    """
    C_true = cost_matrix_numpy(scenario.true_rul[:, None].astype(float), scenario.demand, scenario.costs)
    return float(C_true[np.arange(scenario.n_machines), service_day].sum())


def failures_under(scenario: Scenario, service_day: np.ndarray) -> int:
    """How many machines really fail within the horizon before being serviced."""
    H = scenario.horizon
    t = scenario.true_rul
    return int(np.sum((t < H) & (service_day.astype(float) > t)))
