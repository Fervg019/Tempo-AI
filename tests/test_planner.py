import itertools

import numpy as np
import pytest

from tempo.planner import (alarm_threshold, cost_matrix_numpy, plan_from_samples, realized_cost,
                           run_to_failure, solve_assignment)
from tempo.simulator import CostModel, make_scenario


def reference_cost(T: float, d: int, demand: np.ndarray, c: CostModel) -> float:
    """Slow, obvious cost of ONE future (failure at time T) for service day d (d == H means none)."""
    H = len(demand)
    if d < H and d <= T:
        return c.planned_repair + c.planned_downtime_per_day * c.service_days * demand[d] \
            + c.wasted_life_per_day * (T - d)
    if T < H:
        k = min(max(int(np.floor(T)), 0), H - 1)
        return c.unplanned_repair + c.unplanned_downtime_per_day * c.unplanned_downtime_days * demand[k]
    return 0.0


def test_cost_matrix_matches_slow_reference():
    rng = np.random.default_rng(0)
    demand = np.array([1.0, 1.0, 1.6, 1.6, 0.6, 0.6, 1.0])
    c = CostModel()
    samples = rng.uniform(0.0, 12.0, size=(3, 40))
    C = cost_matrix_numpy(samples, demand, c)
    assert C.shape == (3, len(demand) + 1)
    for i in range(3):
        for d in range(len(demand) + 1):
            expected = np.mean([reference_cost(t, d, demand, c) for t in samples[i]])
            assert C[i, d] == pytest.approx(expected, rel=1e-9)


def test_assignment_is_optimal_and_respects_limits():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n, H = 4, 3
        C = rng.uniform(0, 100, size=(n, H + 1))
        tech = rng.integers(0, 3, size=H)
        parts = rng.integers(0, H, size=n)
        day = solve_assignment(C, tech, parts)

        def feasible(choice):
            for d in range(H):
                if sum(1 for x in choice if x == d) > tech[d]:
                    return False
            return all(x == H or x >= parts[i] for i, x in enumerate(choice))

        assert feasible(day)
        best = min(sum(C[i, x] for i, x in enumerate(ch))
                   for ch in itertools.product(range(H + 1), repeat=n) if feasible(ch))
        assert sum(C[i, day[i]] for i in range(n)) == pytest.approx(best)


def test_plan_never_books_a_day_without_technicians_or_before_parts():
    for seed in range(5):
        sc = make_scenario(seed, n_machines=15, horizon=21)
        rng = np.random.default_rng(seed)
        samples = rng.uniform(0, 40, size=(15, 300))
        plan = plan_from_samples(samples, sc.demand, sc.technicians, sc.parts_ready_day, sc.costs)
        for d in range(sc.horizon):
            assert (plan.service_day == d).sum() <= sc.technicians[d]
        booked = plan.service_day < sc.horizon
        assert (plan.service_day[booked] >= sc.parts_ready_day[booked]).all()
        assert ((plan.p_fail_before_service >= 0) & (plan.p_fail_before_service <= 1)).all()


def test_knowing_the_truth_is_never_worse_than_any_other_valid_plan():
    """With perfect information the planner is optimal, so no valid policy can beat it."""
    for seed in range(5):
        sc = make_scenario(seed, n_machines=12, horizon=21)
        oracle = plan_from_samples(sc.true_rul[:, None].astype(float), sc.demand, sc.technicians,
                                   sc.parts_ready_day, sc.costs).service_day
        for other in (run_to_failure(sc), alarm_threshold(sc, sc.true_rul, 30.0)):
            assert realized_cost(sc, oracle) <= realized_cost(sc, other) + 1e-6


def test_alarm_policy_respects_technicians_and_parts():
    sc = make_scenario(3, n_machines=20, horizon=30)
    day = alarm_threshold(sc, sc.true_rul, 1e9)  # alarm fires for everyone
    for d in range(sc.horizon):
        assert (day == d).sum() <= sc.technicians[d]
    booked = day < sc.horizon
    assert (day[booked] >= sc.parts_ready_day[booked]).all()


def test_torch_matches_numpy_if_torch_is_installed():
    """Skipped on machines without PyTorch. Run it on the AMD GPU machine: it proves the GPU path gives the same answer."""
    torch = pytest.importorskip("torch")
    from tempo.planner import cost_matrix_torch

    rng = np.random.default_rng(2)
    demand = np.array([1.0, 1.0, 1.6, 1.6, 0.6, 0.6, 1.0, 1.0])
    samples = rng.uniform(0.0, 15.0, size=(6, 500))
    a = cost_matrix_numpy(samples, demand, CostModel())
    b = cost_matrix_torch(samples, demand, CostModel(), device="auto")
    np.testing.assert_allclose(a, b, rtol=1e-3)
    assert torch is not None
