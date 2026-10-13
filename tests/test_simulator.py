import numpy as np

from tempo.simulator import SENSOR_NAMES, generate_run_to_failure, make_scenario


def test_same_seed_gives_same_data():
    a = generate_run_to_failure(5, seed=3)
    b = generate_run_to_failure(5, seed=3)
    assert a.equals(b)


def test_rul_counts_down_one_day_at_a_time_and_ends_at_one():
    df = generate_run_to_failure(5, seed=1)
    for _, m in df.groupby("machine_id"):
        assert (np.diff(m["rul"].to_numpy()) == -1).all()
        assert m["rul"].iloc[-1] == 1
        assert (m["health"] > 0).all()


def test_sensors_drift_up_as_machine_wears_out():
    df = generate_run_to_failure(30, seed=2)
    for s in ("vibration", "temperature", "oil_particles"):
        early = df[df["rul"] > 60][s].mean()
        late = df[df["rul"] <= 5][s].mean()
        assert late > early
    assert all(s in df.columns for s in SENSOR_NAMES)


def test_scenario_shapes_and_hidden_truth():
    sc = make_scenario(seed=4, n_machines=12, horizon=21)
    assert sc.n_machines == 12
    assert len(sc.demand) == len(sc.technicians) == 21
    assert len(sc.parts_ready_day) == len(sc.true_rul) == 12
    assert (sc.true_rul >= 1).all()
    assert sc.histories["machine_id"].nunique() == 12
    # the history only goes up to "today": the true failure is still in the future
    last_day = sc.histories.groupby("machine_id")["day"].max().to_numpy()
    assert (sc.true_rul >= 1).all() and (last_day >= 10).all()


def test_scarcity_options_are_respected():
    sc = make_scenario(seed=7, n_machines=25, horizon=28, tech_weekday=1, tech_weekend=0, parts_delay_prob=1.0)
    assert sc.technicians.max() <= 1
    weekend = (np.arange(28) % 7) >= 5
    assert (sc.technicians[weekend] == 0).all()
    assert (sc.parts_ready_day >= 1).all()          # every part is late when the delay chance is 100%

    sc2 = make_scenario(seed=7, n_machines=25, horizon=28, parts_delay_prob=0.0)
    assert (sc2.parts_ready_day == 0).all()          # no part is late when the delay chance is 0%
