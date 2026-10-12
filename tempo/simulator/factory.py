"""Synthetic factory: machines that wear out, noisy sensors, and a fleet scenario.

Everything here is SIMULATED. It exists because public datasets contain sensor
data but no production schedules, technician shifts, spare parts or costs, and
the planner needs all of those to make a decision. Say so openly in the pitch.

How a machine works
-------------------
Each machine has a hidden "health" number that starts at 1.0 and drops a little
every day (random wear, and wear speeds up as health falls). When health hits 0
the machine fails. We never show health to the forecaster; it only sees the
sensors, which are noisy readings that drift as health drops.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SENSOR_NAMES = ["vibration", "temperature", "pressure", "current", "noise_db", "oil_particles"]

# Fixed "physics" shared by all machines (same sensor behaves the same everywhere).
_BASE = np.array([2.0, 60.0, 5.0, 12.0, 70.0, 5.0])    # healthy reading
_GAIN = np.array([4.0, 25.0, 0.3, 6.0, 12.0, 30.0])    # how much it rises when worn out
_POWER = np.array([2.0, 1.5, 1.0, 1.0, 1.0, 3.0])      # shape: >1 means "only rises late"
_NOISE = np.array([0.5, 2.5, 0.4, 0.8, 1.5, 1.5])      # random reading noise


@dataclass(frozen=True)
class DegradationConfig:
    mean_life_days: float = 90.0   # typical nominal life
    life_sigma: float = 0.35       # how different machines are from each other (lognormal)
    shape: float = 2.0             # lower = wear is jumpier day to day
    acceleration: float = 1.5      # wear speeds up as the machine gets worse
    noise_scale: float = 1.0       # multiply sensor noise (make the problem easier/harder)
    max_days: int = 600


def simulate_machine(rng: np.random.Generator, cfg: DegradationConfig, machine_id: int) -> pd.DataFrame:
    """One machine from install day until failure. Columns include true health and rul."""
    life = cfg.mean_life_days * np.exp(rng.normal(0.0, cfg.life_sigma))
    base_rate = 1.0 / life
    offsets = rng.normal(0.0, 1.0, size=len(SENSOR_NAMES)) * _NOISE  # each machine reads a bit differently

    health: list[float] = []
    h = 1.0
    day = 0
    while h > 0.0 and day < cfg.max_days:
        health.append(h)
        mean_wear = base_rate * (1.0 + cfg.acceleration * (1.0 - h))
        h -= rng.gamma(cfg.shape, mean_wear / cfg.shape)
        day += 1
    fail_day = day  # the machine fails on this day index
    health_arr = np.array(health)
    wear = np.clip(1.0 - health_arr, 0.0, 1.0)[:, None]

    readings = (
        _BASE + offsets + _GAIN * wear ** _POWER
        + rng.normal(0.0, 1.0, size=(len(health_arr), len(SENSOR_NAMES))) * _NOISE * cfg.noise_scale
    )
    df = pd.DataFrame(readings, columns=SENSOR_NAMES)
    df.insert(0, "day", np.arange(len(health_arr)))
    df.insert(0, "machine_id", machine_id)
    df["health"] = health_arr          # hidden truth, never a model input
    df["rul"] = fail_day - df["day"]   # true remaining days until failure (>= 1)
    return df


def generate_run_to_failure(n_machines: int, seed: int = 0, cfg: DegradationConfig | None = None,
                            first_id: int = 0) -> pd.DataFrame:
    """Many machines run until they fail. Used to train and test the forecaster."""
    cfg = cfg or DegradationConfig()
    rng = np.random.default_rng(seed)
    frames = [simulate_machine(rng, cfg, first_id + i) for i in range(n_machines)]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Fleet scenario: the situation the planner must make a decision about.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """All money numbers are assumptions. Show them openly and test other values."""
    unplanned_downtime_per_day: float = 20_000.0  # line stops without warning (scaled by demand)
    unplanned_repair: float = 8_000.0             # emergency repair after a failure
    planned_downtime_per_day: float = 3_000.0     # planned stop, work is re-routed (scaled by demand)
    planned_repair: float = 2_000.0               # normal repair
    wasted_life_per_day: float = 150.0            # servicing early throws away remaining life
    service_days: int = 1                         # how long a repair takes


@dataclass
class Scenario:
    seed: int
    horizon: int                         # days we plan over
    histories: pd.DataFrame              # sensor history of every machine up to "today"
    machine_ids: np.ndarray              # (n,)
    demand: np.ndarray                   # (horizon,) production demand multiplier per day
    technicians: np.ndarray              # (horizon,) technicians on shift per day
    parts_ready_day: np.ndarray          # (n,) first day the spare part is available
    costs: CostModel
    true_rul: np.ndarray = field(repr=False)  # (n,) TRUE days left; used only to score policies

    @property
    def n_machines(self) -> int:
        return len(self.machine_ids)


def make_scenario(seed: int, n_machines: int = 20, horizon: int = 30,
                  cfg: DegradationConfig | None = None, costs: CostModel | None = None) -> Scenario:
    """A fleet where every machine is somewhere in its life, plus schedule, staff and parts."""
    cfg = cfg or DegradationConfig()
    costs = costs or CostModel()
    rng = np.random.default_rng(seed)

    frames, true_rul = [], []
    for mid in range(n_machines):
        full = simulate_machine(rng, cfg, mid)
        life = len(full)
        # "today" is somewhere between 35% and 95% of its life, so some are healthy and some are close to failing
        today = int(np.clip(rng.uniform(0.35, 0.95) * life, 10, life - 1))
        frames.append(full.iloc[: today + 1])
        true_rul.append(float(life - today))
    histories = pd.concat(frames, ignore_index=True)

    days = np.arange(horizon)
    weekend = (days % 7) >= 5
    demand = np.where(weekend, 0.6, 1.0)
    for _ in range(2):  # two busy periods (big orders)
        start = int(rng.integers(0, max(1, horizon - 4)))
        demand[start:start + 4] *= 1.6
    technicians = np.where(weekend, 1, 2).astype(int)
    technicians[rng.integers(0, horizon)] = 0  # one day nobody is available

    parts_ready = np.where(rng.random(n_machines) < 0.6, 0, rng.integers(1, max(2, horizon // 2), n_machines))

    return Scenario(
        seed=seed, horizon=horizon, histories=histories, machine_ids=np.arange(n_machines),
        demand=demand, technicians=technicians, parts_ready_day=parts_ready.astype(int),
        costs=costs, true_rul=np.array(true_rul),
    )
