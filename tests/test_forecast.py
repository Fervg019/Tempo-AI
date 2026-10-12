import numpy as np
import pytest

from tempo.forecast import QuantileForecaster, evaluate, latest_rows
from tempo.simulator import generate_run_to_failure, make_scenario


@pytest.fixture(scope="module")
def trained():
    train = generate_run_to_failure(120, seed=0)
    test = generate_run_to_failure(40, seed=999, first_id=10_000)
    model = QuantileForecaster(max_iter=100).fit(train)
    return model, test


def test_beats_guessing_the_average(trained):
    model, test = trained
    m = evaluate(model, test)
    assert m["all_mae_days"] < 0.7 * m["naive_mae_days"]


def test_ranges_are_roughly_honest(trained):
    model, test = trained
    m = evaluate(model, test)
    assert 0.70 < m["all_coverage_90"] < 0.99


def test_quantiles_are_ordered_and_non_negative(trained):
    model, test = trained
    q = model.predict_quantiles(test)
    assert (np.diff(q, axis=1) >= 0).all()
    assert (q >= 0).all()


def test_sampling_stays_inside_modelled_range(trained):
    model, _ = trained
    sc = make_scenario(seed=5, n_machines=8, horizon=21)
    q = model.predict_from_features(latest_rows(sc.histories))
    assert q.shape[0] == 8
    draws = model.sample_rul(q, n_samples=500, rng=np.random.default_rng(0))
    assert draws.shape == (8, 500)
    assert (draws >= q[:, [0]] - 1e-9).all() and (draws <= q[:, [-1]] + 1e-9).all()
