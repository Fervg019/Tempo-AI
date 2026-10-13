"""Expected cost of servicing each machine on each day.

The idea
--------
We do not know exactly when a machine will fail, but the forecaster gives a
range. We "roll the dice" S times (thousands) to draw possible failure times
for every machine. For every machine i and every candidate service day d we then
ask: across all those possible futures, what is the AVERAGE cost?

    C[i, d]  for d = 0 .. H-1   -> service machine i at the start of day d
    C[i, H]                     -> do not service it within the horizon

The tensor work is n_machines x n_days x n_samples. That is cheap for 20 machines,
but a fleet of thousands of machines with tens of thousands of samples is billions
of numbers, which is exactly the kind of job a GPU is good at.

Cost of one possible future (failure at time T days from now)
-------------------------------------------------------------
service on day d, and d <= T (we get there in time):
    planned repair + planned downtime on that day (busy days cost more)
    + wasted life (we threw away T - d days of remaining life)
otherwise (it fails first, or we never service) and T < H:
    emergency repair + longer unplanned downtime (on the day it fails)
otherwise (no service and it survives the horizon): 0

Two implementations with identical results: NumPy (works anywhere, including
a locked-down laptop) and PyTorch (runs on CPU or an AMD/ROCm GPU).
"""
from __future__ import annotations

import numpy as np

from tempo.simulator import CostModel


def cost_matrix_numpy(samples: np.ndarray, demand: np.ndarray, costs: CostModel,
                      chunk_elems: int = 40_000_000) -> np.ndarray:
    """samples: (n, S) sampled failure times in days. Returns (n, H+1) expected costs."""
    n, S = samples.shape
    H = len(demand)
    days = np.arange(H, dtype=np.float64)[None, :, None]
    dem_d = demand[None, :, None]
    out = np.zeros((n, H + 1))
    chunk = max(1, chunk_elems // max(1, n * H))
    for a in range(0, S, chunk):
        T = samples[:, a:a + chunk].astype(np.float64)                     # (n, s)
        idx = np.clip(np.floor(T).astype(int), 0, H - 1)
        unplanned = np.where(
            T < H,
            costs.unplanned_repair + costs.unplanned_downtime_per_day * costs.unplanned_downtime_days * demand[idx],
            0.0,
        )                                                                    # (n, s)
        planned = (costs.planned_repair
                   + costs.planned_downtime_per_day * costs.service_days * dem_d
                   + costs.wasted_life_per_day * (T[:, None, :] - days))     # (n, H, s)
        c = np.where(days <= T[:, None, :], planned, unplanned[:, None, :])
        out[:, :H] += c.sum(axis=2)
        out[:, H] += unplanned.sum(axis=1)
    return out / S


def cost_matrix_torch(samples, demand, costs: CostModel, device: str = "auto",
                      chunk_elems: int = 200_000_000) -> np.ndarray:
    """Same maths as cost_matrix_numpy, on a PyTorch device. Returns a NumPy array (n, H+1)."""
    import torch

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    n, S = samples.shape
    H = len(demand)
    dem = torch.as_tensor(demand, dtype=torch.float32, device=dev)
    days = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    dem_d = dem.view(1, H, 1)
    out = torch.zeros((n, H + 1), dtype=torch.float64, device=dev)
    chunk = max(1, chunk_elems // max(1, n * H))
    for a in range(0, S, chunk):
        T = torch.as_tensor(np.ascontiguousarray(samples[:, a:a + chunk]), dtype=torch.float32, device=dev)
        idx = torch.floor(T).long().clamp(0, H - 1)
        unplanned = torch.where(
            T < H,
            costs.unplanned_repair + costs.unplanned_downtime_per_day * costs.unplanned_downtime_days * dem[idx],
            torch.zeros_like(T),
        )
        planned = (costs.planned_repair
                   + costs.planned_downtime_per_day * costs.service_days * dem_d
                   + costs.wasted_life_per_day * (T[:, None, :] - days))
        c = torch.where(days <= T[:, None, :], planned, unplanned[:, None, :])
        out[:, :H] += c.sum(dim=2, dtype=torch.float64)
        out[:, H] += unplanned.sum(dim=1, dtype=torch.float64)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (out / S).cpu().numpy()
