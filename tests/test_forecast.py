import numpy as np
import pytest

from tempo.forecast import CalibratedForecaster, QuantileForecaster, evaluate, latest_rows
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


def test_sampling_without_tails_stays_inside_modelled_range(trained):
    model, _ = trained
    sc = make_scenario(seed=5, n_machines=8, horizon=21)
    q = model.predict_from_features(latest_rows(sc.histories))
    assert q.shape[0] == 8
    draws = model.sample_rul(q, n_samples=500, rng=np.random.default_rng(0), tails=False)
    assert draws.shape == (8, 500)
    assert (draws >= q[:, [0]] - 1e-9).all() and (draws <= q[:, [-1]] + 1e-9).all()


def test_sampling_with_tails_includes_riskier_early_failures(trained):
    """The default sampler also draws from below the lowest forecast quantile (the riskiest futures)."""
    model, _ = trained
    sc = make_scenario(seed=5, n_machines=8, horizon=21)
    q = model.predict_from_features(latest_rows(sc.histories))
    draws = model.sample_rul(q, n_samples=4000, rng=np.random.default_rng(0))
    assert (draws >= 0).all() and (draws <= model.rul_cap + 1e-9).all()
    below = (draws < q[:, [0]] - 1e-9).mean()
    assert 0.02 < below < 0.09          # about 5% of futures sit below the 5% forecast


def test_calibration_keeps_quantiles_valid_and_ranges_reasonable(trained):
    model, test = trained
    cal = CalibratedForecaster(model).fit(generate_run_to_failure(80, seed=500, first_id=50_000))
    q = cal.predict_quantiles(test)
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    m = evaluate(cal, test)
    assert 0.80 < m["all_coverage_90"] < 0.98
    assert 0.80 < m["late_coverage_90"] < 0.98
