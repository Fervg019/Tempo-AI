"""Baseline remaining-useful-life (RUL) forecaster that returns a RANGE, not one number.

Method: one gradient-boosting model per quantile (5%, 10%, 25%, 50%, 75%, 90%, 95%).
"The 5% value is 8 days and the 95% value is 21 days" means: we expect the
machine to last between 8 and 21 days, 90% of the time. The planner needs that
range to weigh risk, and we check it is honest ("calibration").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import build_features, feature_columns

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def sample_from_quantiles(quantile_values: np.ndarray, quantiles, n_samples: int, rng: np.random.Generator,
                          rul_cap: float, tails: bool = True) -> np.ndarray:
    """Turn forecast quantiles into random failure times (days from now), one row per machine."""
    qs = np.asarray(quantiles, dtype=float)
    qv = np.asarray(quantile_values, dtype=float)
    if tails:
        # Extend the curve to levels 0 and 1 using the slope of the outermost interval.
        lo = np.maximum(0.0, qv[:, :1] - (qv[:, 1:2] - qv[:, :1]) * (qs[0] / (qs[1] - qs[0])))
        hi = np.minimum(rul_cap, qv[:, -1:] + (qv[:, -1:] - qv[:, -2:-1]) * ((1 - qs[-1]) / (qs[-1] - qs[-2])))
        qs = np.concatenate([[0.0], qs, [1.0]])
        qv = np.concatenate([lo, qv, hi], axis=1)
    u = rng.uniform(qs[0], qs[-1], size=(qv.shape[0], n_samples))
    return np.stack([np.interp(u[i], qs, qv[i]) for i in range(qv.shape[0])])


class QuantileForecaster:
    def __init__(self, rul_cap: float = 120.0, quantiles=QUANTILES, max_iter: int = 200,
                 learning_rate: float = 0.08, max_leaf_nodes: int = 15, seed: int = 0):
        # RUL far from failure is basically unknowable, so we cap the target.
        self.rul_cap = rul_cap
        self.quantiles = tuple(quantiles)
        self._params = dict(max_iter=max_iter, learning_rate=learning_rate,
                            max_leaf_nodes=max_leaf_nodes, random_state=seed)
        self.models: list[HistGradientBoostingRegressor] = []
        self.columns: list[str] = []

    # ---- training ----
    def fit(self, df: pd.DataFrame) -> "QuantileForecaster":
        """df: run-to-failure data with a true 'rul' column."""
        df = df.sort_values(["machine_id", "day"]).reset_index(drop=True)
        feats = build_features(df)
        self.columns = feature_columns(feats)
        y = np.minimum(df["rul"].to_numpy(dtype=float), self.rul_cap)
        self.models = [
            HistGradientBoostingRegressor(loss="quantile", quantile=q, **self._params).fit(feats[self.columns], y)
            for q in self.quantiles
        ]
        return self

    # ---- prediction ----
    def predict_from_features(self, feats: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_quantiles), sorted so higher quantiles are never lower, never below 0."""
        out = np.column_stack([m.predict(feats[self.columns]) for m in self.models])
        return np.clip(np.sort(out, axis=1), 0.0, self.rul_cap)

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        feats = build_features(df)
        return self.predict_from_features(feats)

    def sample_rul(self, quantile_values: np.ndarray, n_samples: int, rng: np.random.Generator,
                   tails: bool = True) -> np.ndarray:
        """Draw possible failure times from each machine's forecast range.

        quantile_values: (n_machines, n_quantiles). Returns (n_machines, n_samples).
        This is the 'roll the dice' step the planner uses. With tails=True (default) the
        lowest 5% and highest 5% of futures are included too, extended in a straight line
        from the outermost forecast values. The riskiest futures (early failures) live in
        that lowest tail, so ignoring it makes the planner too relaxed.
        """
        return sample_from_quantiles(quantile_values, self.quantiles, n_samples, rng, self.rul_cap, tails)


def evaluate(model: QuantileForecaster, df: pd.DataFrame, late_life_days: float = 30.0) -> dict[str, float]:
    """Accuracy and calibration. 'late' = machines within `late_life_days` of failing (where it matters most)."""
    df = df.sort_values(["machine_id", "day"]).reset_index(drop=True)
    pred = model.predict_quantiles(df)
    y = np.minimum(df["rul"].to_numpy(dtype=float), model.rul_cap)
    q = list(model.quantiles)
    med = pred[:, q.index(0.50)]
    lo90, hi90 = pred[:, q.index(0.05)], pred[:, q.index(0.95)]
    lo50, hi50 = pred[:, q.index(0.25)], pred[:, q.index(0.75)]
    late = y <= late_life_days

    def stats(mask: np.ndarray) -> dict[str, float]:
        return {
            "mae_days": float(np.mean(np.abs(med[mask] - y[mask]))),
            "coverage_90": float(np.mean((y[mask] >= lo90[mask]) & (y[mask] <= hi90[mask]))),
            "coverage_50": float(np.mean((y[mask] >= lo50[mask]) & (y[mask] <= hi50[mask]))),
            "width_90_days": float(np.mean(hi90[mask] - lo90[mask])),
            "n": int(mask.sum()),
        }

    all_stats, late_stats = stats(np.ones_like(y, dtype=bool)), stats(late)
    naive_mae = float(np.mean(np.abs(np.mean(y) - y)))  # "always guess the average" baseline
    return {
        **{f"all_{k}": v for k, v in all_stats.items()},
        **{f"late_{k}": v for k, v in late_stats.items()},
        "naive_mae_days": naive_mae,
    }
