import numpy as np
import pytest

from tempo.planner import plan_from_samples
from tempo.simulator import make_scenario
from tempo.validator import (Policy, check_numbers, facts_from_plan, validate_plan, validate_work_order)


def _scenario_and_plan(seed=0, n=12, horizon=21):
    sc = make_scenario(seed, n_machines=n, horizon=horizon)
    samples = np.random.default_rng(seed).uniform(0, 40, size=(n, 400))
    plan = plan_from_samples(samples, sc.demand, sc.technicians, sc.parts_ready_day, sc.costs)
    return sc, plan


GOOD_ORDER = {
    "machine_id": 0, "service_day": 0,
    "steps": ["Inspect bearings", "Replace worn parts", "Function test before handing back"],
    "safety_steps": ["Lockout/tagout the main breaker", "Isolate all energy sources", "Wear PPE (gloves, safety glasses)"],
}


def test_planner_output_passes_the_rulebook():
    for seed in range(6):
        sc, plan = _scenario_and_plan(seed)
        rep = validate_plan(sc, plan.service_day, plan.p_fail_before_service)
        assert rep.ok, rep.summary()


def test_catches_repair_on_a_day_nobody_is_on_shift():
    sc, plan = _scenario_and_plan(1)
    day = plan.service_day.copy()
    zero_day = int(np.where(sc.technicians == 0)[0][0])
    day[0] = zero_day
    rep = validate_plan(sc, day)
    assert "TECHNICIANS" in rep.rules_failed() or "PARTS" in rep.rules_failed()
    assert not rep.ok


def test_catches_too_many_repairs_in_one_day():
    sc = make_scenario(2, n_machines=10, horizon=14)
    sc.parts_ready_day[:] = 0
    day = np.full(10, sc.horizon)
    d = int(np.where(sc.technicians == sc.technicians.max())[0][0])
    day[: sc.technicians[d] + 1] = d            # one more repair than technicians
    rep = validate_plan(sc, day)
    assert "TECHNICIANS" in rep.rules_failed()


def test_catches_repair_before_the_part_arrives():
    sc = make_scenario(3, n_machines=6, horizon=21)
    sc.parts_ready_day[:] = 0
    sc.parts_ready_day[2] = 5
    day = np.full(6, sc.horizon)
    d = int(np.where(sc.technicians > 0)[0][0])
    assert d < 5
    day[2] = d
    rep = validate_plan(sc, day)
    assert "PARTS" in rep.rules_failed()


def test_catches_days_out_of_range_and_wrong_length():
    sc, plan = _scenario_and_plan(4)
    bad = plan.service_day.copy()
    bad[1] = sc.horizon + 3
    assert "RANGE" in validate_plan(sc, bad).rules_failed()
    assert "SHAPE" in validate_plan(sc, plan.service_day[:-1]).rules_failed()


def test_company_daily_cap_is_enforced():
    sc = make_scenario(5, n_machines=8, horizon=14, tech_weekday=5, tech_weekend=5)
    sc.technicians[:] = 5
    sc.parts_ready_day[:] = 0
    day = np.full(8, sc.horizon)
    day[:4] = 3
    assert validate_plan(sc, day).ok
    assert "DAILY_CAP" in validate_plan(sc, day, policy=Policy(max_machines_per_day=2)).rules_failed()


def test_risky_unrepaired_machine_is_a_warning_not_an_error():
    sc = make_scenario(6, n_machines=5, horizon=14)
    day = np.full(5, sc.horizon)
    p = np.array([0.9, 0.0, 0.0, 0.0, 0.0])
    rep = validate_plan(sc, day, p)
    assert rep.ok and len(rep.warnings) == 1 and rep.warnings[0].rule == "RISK"


def test_good_work_order_passes():
    sc, plan = _scenario_and_plan(0)
    day = plan.service_day.copy()
    d = int(np.where(sc.technicians > 0)[0][0])
    day[0] = d
    order = dict(GOOD_ORDER, service_day=d)
    assert validate_work_order(order, sc, day).ok


def test_work_order_that_changes_the_day_is_rejected():
    sc, plan = _scenario_and_plan(0)
    day = plan.service_day.copy()
    d = int(np.where(sc.technicians > 0)[0][0])
    day[0] = d
    rep = validate_work_order(dict(GOOD_ORDER, service_day=d + 1), sc, day)
    assert "ORDER_DAY" in rep.rules_failed()


def test_work_order_missing_safety_steps_is_rejected():
    sc, plan = _scenario_and_plan(0)
    day = plan.service_day.copy()
    d = int(np.where(sc.technicians > 0)[0][0])
    day[0] = d
    order = dict(GOOD_ORDER, service_day=d, safety_steps=["Wear PPE"],
                 steps=["Replace the bearing"])
    rep = validate_work_order(order, sc, day)
    failed = [f.message for f in rep.errors if f.rule == "SAFETY"]
    assert any("lockout" in m for m in failed) and any("energy isolation" in m for m in failed)


def test_all_problems_in_a_work_order_are_reported_together():
    sc, plan = _scenario_and_plan(0)
    day = plan.service_day.copy()
    d = int(np.where(sc.technicians > 0)[0][0])
    day[0] = d
    order = dict(GOOD_ORDER, service_day=d + 1, steps=["Replace the bearing"], safety_steps=["Wear PPE"])
    failed = validate_work_order(order, sc, day).rules_failed()
    assert {"ORDER_DAY", "SAFETY"} <= failed


def test_work_order_format_problems_are_caught():
    sc, plan = _scenario_and_plan(0)
    assert "ORDER_FORMAT" in validate_work_order({"machine_id": 0}, sc, plan.service_day).rules_failed()
    assert "ORDER_FORMAT" in validate_work_order("not json", sc, plan.service_day).rules_failed()
    assert "ORDER_MACHINE" in validate_work_order(dict(GOOD_ORDER, machine_id=999), sc,
                                                  plan.service_day).rules_failed()


def test_work_order_for_a_machine_with_no_repair_is_rejected():
    sc = make_scenario(7, n_machines=4, horizon=14)
    day = np.full(4, sc.horizon)
    assert "ORDER_UNPLANNED" in validate_work_order(dict(GOOD_ORDER, machine_id=1), sc, day).rules_failed()


def test_number_check_accepts_real_numbers_and_flags_invented_ones():
    sc, plan = _scenario_and_plan(0)
    facts = facts_from_plan(sc, plan.service_day, plan.p_fail_before_service, plan.expected_cost,
                            plan.expected_cost_by_machine)
    i = int(np.argmax(plan.expected_cost_by_machine))
    cost = plan.expected_cost_by_machine[i]
    p = 100 * plan.p_fail_before_service[i]
    honest = f"Machine {i} has an expected cost of ${cost:,.0f} and a {p:.0f}% chance of failing early."
    assert check_numbers(honest, facts).findings == []

    made_up = honest + " Servicing it saves $987,654 and avoids a 97.3% failure risk."
    rep = check_numbers(made_up, facts)
    flagged = " ".join(f.message for f in rep.warnings)
    assert "987,654" in flagged
    assert len(rep.warnings) >= 1
    assert rep.ok          # unverified numbers are warnings: a human reviews them
