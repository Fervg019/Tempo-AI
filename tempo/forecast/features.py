"""Turn raw sensor history into features a model can learn from.

Why: one sensor reading says little. What matters is the trend (rising over the
last days), how noisy it is, and how far it has moved from the machine's own
"healthy" level. Each machine reads a bit differently, so we compare a machine
to itself.
"""
from __future__ import annotations

import pandas as pd

from tempo.simulator import SENSOR_NAMES

WINDOWS = (5, 15)
BASELINE_DAYS = 5  # the first days after install define "healthy" for that machine


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs machine_id, day and the sensor columns. Returns one feature row per input row.

    Only past and present readings are used for each row (no peeking ahead),
    so the same function works on a live history.
    """
    df = df.sort_values(["machine_id", "day"]).reset_index(drop=True)
    g = df.groupby("machine_id", sort=False)

    cols: dict[str, pd.Series] = {"machine_id": df["machine_id"], "day": df["day"], "age": df["day"]}
    for s in SENSOR_NAMES:
        x = df[s]
        baseline = g[s].transform(lambda v: v.iloc[:BASELINE_DAYS].mean())
        cols[s] = x
        cols[f"{s}_delta0"] = x - baseline                      # change since the machine was new
        cols[f"{s}_diff5"] = g[s].diff(5).fillna(0.0)           # short-term trend
        for w in WINDOWS:
            roll = g[s].rolling(w, min_periods=1)
            cols[f"{s}_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
            cols[f"{s}_std{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
    return pd.DataFrame(cols)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("machine_id", "day")]


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Features for the most recent day of each machine ('today' in a live scenario)."""
    feats = build_features(df)
    return feats.sort_values(["machine_id", "day"]).groupby("machine_id", sort=True).tail(1).reset_index(drop=True)
