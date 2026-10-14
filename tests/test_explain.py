import numpy as np
import pytest

from tempo.forecast import QuantileForecaster, latest_rows
from tempo.llm import PROCEDURES, FakeLLM, LLMError, build_facts, explain_plan, template_work_order
from tempo.llm.explain import allowed_numbers
from tempo.planner import plan_fleet
from tempo.simulator import generate_run_to_failure, make_scenario
from tempo.validator import check_numbers, validate_work_order


@pytest.fixture(scope="module")
def setup():
    model = QuantileForecaster(max_iter=60).fit(generate_run_to_failure(80, seed=0))
    sc = make_scenario(11, n_machines=12, horizon=21, tech_weekday=1, tech_weekend=0)
    feats = latest_rows(sc.histories)
    q = model.predict_from_features(feats)
    plan = plan_fleet(sc, model, q, n_samples=800, seed=1)
    assert (plan.service_day < sc.horizon).sum() >= 4          # the tests below need real work orders
    return sc, plan, q, feats


class Broken:
    calls = 0

    def chat_json(self, *a, **k):
        raise LLMError("server down")


class Stubborn(FakeLLM):
    """Keeps making the same mistakes even when told what was wrong."""

    def _orders(self, payload):
        payload = dict(payload)
        payload.pop("fix_these_problems", None)
        return super()._orders(payload)


def all_valid(result, sc, plan):
    return all(validate_work_order(o, sc, plan.service_day).ok for o in result.work_orders.values())


def test_good_llm_passes_everything_first_try(setup):
    sc, plan, q, feats = setup
    r = explain_plan(FakeLLM("good"), sc, plan, q, feats)
    assert r.plan_report.ok and all_valid(r, sc, plan)
    assert set(r.order_source.values()) == {"llm"}
    assert r.stats["first_try_valid_rate"] == 1.0 and r.stats["unverified_numbers_in_text"] == 0
    assert len(r.work_orders) == int((plan.service_day < sc.horizon).sum())


def test_sloppy_llm_is_caught_and_repaired(setup):
    sc, plan, q, feats = setup
    r = explain_plan(FakeLLM("sloppy"), sc, plan, q, feats)
    assert all_valid(r, sc, plan)                               # nothing invalid ever reaches a human
    assert r.stats["first_try_valid_rate"] < 1.0 and r.stats["problems_caught"] > 0
    assert r.stats["llm_valid_after_repair"] > 0
    assert r.stats["unverified_numbers_in_text"] >= 1           # the invented $987,654
    assert any("987,654" in f.message for f in r.explanation_report.warnings)


def test_stubborn_llm_falls_back_to_templates(setup):
    sc, plan, q, feats = setup
    r = explain_plan(Stubborn("sloppy"), sc, plan, q, feats)
    assert all_valid(r, sc, plan)
    assert r.stats["template_fallback"] > 0
    assert "template" in r.order_source.values()


def test_dead_llm_still_gives_a_valid_plan_summary_and_orders(setup):
    sc, plan, q, feats = setup
    r = explain_plan(Broken(), sc, plan, q, feats)
    assert all_valid(r, sc, plan)
    assert set(r.order_source.values()) == {"template"}
    assert "machines have a repair planned" in r.summary
    assert len(r.reasons) == int((plan.service_day < sc.horizon).sum())


def test_batching_keeps_api_calls_low(setup):
    sc, plan, q, feats = setup
    r = explain_plan(FakeLLM("good"), sc, plan, q, feats, batch_size=6, max_orders=8)
    assert r.llm_calls <= 3                                     # 1 explanation + at most 2 order batches


def test_facts_contain_only_real_numbers_and_no_repair_fields_for_unscheduled(setup):
    sc, plan, q, feats = setup
    facts = build_facts(sc, plan, q, feats)
    assert len(facts["machines"]) == sc.n_machines
    for m in facts["machines"]:
        i = m["machine_id"]
        if plan.service_day[i] >= sc.horizon:
            assert m["service_day"] is None and "technicians_on_that_day" not in m
        else:
            assert m["service_day"] == int(plan.service_day[i])
    text = " ".join(f"{m['chance_of_failing_before_repair_percent']}% day {m['service_day']}"
                    for m in facts["machines"] if m["service_day"] is not None)
    assert check_numbers(text, allowed_numbers(sc, plan, facts)).findings == []


@pytest.mark.parametrize("sensor", list(PROCEDURES))
def test_template_orders_always_pass_the_rulebook(sensor, setup):
    sc, plan, _, _ = setup
    day = np.full(sc.n_machines, sc.horizon)
    d = int(np.where(sc.technicians > 0)[0][0])
    day[0] = d
    assert validate_work_order(template_work_order(0, d, sensor), sc, day).ok


class Spy(FakeLLM):
    """Records every message sent to the model."""

    def __init__(self, mode="good"):
        super().__init__(mode)
        self.seen = []

    def chat_json(self, messages, *a, **k):
        self.seen.append(messages)
        return super().chat_json(messages, *a, **k)


def test_safety_rules_are_in_the_order_prompt_only_when_asked(setup):
    sc, plan, q, feats = setup
    told, bare = Spy("good"), Spy("good")
    explain_plan(told, sc, plan, q, feats)
    r = explain_plan(bare, sc, plan, q, feats, tell_model_the_rules=False)

    def order_prompts(spy):
        return [m for m in spy.seen if "work orders" in m[0]["content"]]

    assert order_prompts(told) and order_prompts(bare)
    assert all("lockout" in m[0]["content"] and "required_safety_steps" in m[-1]["content"] for m in order_prompts(told))
    assert all("lockout" not in m[0]["content"] and "required_safety_steps" not in m[-1]["content"]
               for m in order_prompts(bare))
    assert r.stats["rules_in_prompt"] == 0.0


class Weird:
    calls = 0

    def chat_json(self, *a, **k):
        return {"summary": 5, "machines": 7, "work_orders": "nope"}


def test_oddly_shaped_model_replies_fall_back_instead_of_crashing(setup):
    sc, plan, q, feats = setup
    r = explain_plan(Weird(), sc, plan, q, feats)
    assert all_valid(r, sc, plan)
    assert set(r.order_source.values()) == {"template"}
    assert "machines have a repair planned" in r.summary


def test_prompts_never_contain_a_concrete_example_id_the_model_could_copy():
    from tempo.llm.explain import EXPLAIN_SYSTEM, ORDER_SYSTEM, ORDER_SYSTEM_BARE
    for prompt in (EXPLAIN_SYSTEM, ORDER_SYSTEM, ORDER_SYSTEM_BARE):
        assert '"machine_id": 3' not in prompt and '"service_day": 5' not in prompt
        assert "<id>" in prompt
