"""Fix overconfident forecast ranges using machines the model never trained on.

The problem: a forecaster trained on one set of machines is a bit too sure of itself on new
ones. Its "90% range" holds the truth only ~81% of the time just before failure.

The fix (a simple form of conformal calibration):
  1. Take extra machines that were NOT used for training (the "calibration set").
  2. For every forecast quantile (5%, 10%, ... 95%) look at the ratio  truth / forecast.
     If the 5% forecast is right, the truth falls below it exactly 5% of the time. If it
     really happens 9% of the time, the forecast is too high and we scale it down until it
     happens 5% of the time.
  3. Do this separately for machines that look close to failure and ones that look healthy,
     because the model is wrong in different ways in each situation. ("Close" is judged by
     the forecast's own median, which is available for any machine at any time.)

The result has the same interface as QuantileForecaster, so the planner can use either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import QuantileForecaster, sample_from_quantiles
from .features import build_features

DEFAULT_EDGES = (8.0, 16.0, 30.0, 60.0)   # boundaries between "bands" of predicted median RUL (days)


class CalibratedForecaster:
    def __init__(self, base: QuantileForecaster, edges=DEFAULT_EDGES, min_rows_per_band: int = 300):
        self.base = base
        self.edges = np.asarray(edges, dtype=float)
        self.min_rows = min_rows_per_band
        self.scales: np.ndarray | None = None   # (n_bands, n_quantiles)

    # Same attributes the planner and evaluate() expect.
    @property
    def quantiles(self):
        return self.base.quantiles

    @property
    def rul_cap(self):
        return self.base.rul_cap

    def _band(self, median: np.ndarray) -> np.ndarray:
        return np.digitize(median, self.edges)

    def fit(self, calibration_df: pd.DataFrame) -> "CalibratedForecaster":
        """calibration_df: run-to-failure data of machines NOT used to train the base model."""
        df = calibration_df.sort_values(["machine_id", "day"]).reset_index(drop=True)
        pred = self.base.predict_from_features(build_features(df))
        y = np.minimum(df["rul"].to_numpy(dtype=float), self.rul_cap)
        med = pred[:, list(self.quantiles).index(0.50)]
        band = self._band(med)
        n_bands = len(self.edges) + 1

        def scales_for(mask: np.ndarray) -> np.ndarray:
            out = np.ones(len(self.quantiles))
            for j, q in enumerate(self.quantiles):
                p = np.maximum(pred[mask, j], 0.5)            # avoid dividing by ~0
                out[j] = np.quantile(y[mask] / p, q)           # scale so truth falls below it q of the time
            return out

        global_scales = scales_for(np.ones(len(y), dtype=bool))
        self.scales = np.stack([
            scales_for(band == b) if (band == b).sum() >= self.min_rows else global_scales
            for b in range(n_bands)
        ])
        return self

    def predict_from_features(self, feats: pd.DataFrame) -> np.ndarray:
        assert self.scales is not None, "call fit() first"
        pred = self.base.predict_from_features(feats)
        band = self._band(pred[:, list(self.quantiles).index(0.50)])
        out = pred * self.scales[band]
        return np.clip(np.sort(out, axis=1), 0.0, self.rul_cap)

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_from_features(build_features(df))

    def sample_rul(self, quantile_values: np.ndarray, n_samples: int, rng: np.random.Generator,
                   tails: bool = True) -> np.ndarray:
        return sample_from_quantiles(quantile_values, self.quantiles, n_samples, rng, self.rul_cap, tails)
