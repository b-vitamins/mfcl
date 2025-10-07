from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _percentiles(vals: List[float], qs=(0.25, 0.5, 0.75)) -> Tuple[float, float, float]:
    t = torch.tensor(vals)
    return (
        float(t.quantile(qs[0]).item()),
        float(t.quantile(qs[1]).item()),
        float(t.quantile(qs[2]).item()),
    )


def summarize_timing(ms_list: List[float]) -> Dict[str, float]:
    q25, med, q75 = _percentiles(ms_list, (0.25, 0.5, 0.75))
    tensor_vals = torch.tensor(ms_list)
    return {
        "ms_p25": q25,
        "ms_median": med,
        "ms_p75": q75,
        "ms_mean": float(tensor_vals.mean().item()),
        "ms_std": float(tensor_vals.std(unbiased=False).item()),
    }


def loglog_slope(xs: List[int], ys_ms: List[float]) -> float:
    x = torch.log(torch.tensor([float(v) for v in xs]))
    y = torch.log(torch.tensor([float(v) for v in ys_ms]))
    A = torch.stack([torch.ones_like(x), x], dim=1)
    sol = torch.linalg.lstsq(A, y).solution
    return float(sol[1].item())


def metrics_row(
    *,
    run_kind: str,
    dataset: str,
    world_size: int,
    device: str,
    dtype: str,
    N_global: int,
    d: int,
    method: str,
    covariance: str,
    shrinkage: float,
    reps: int,
    warmup: int,
    ms_stats: Dict[str, float],
    peak_mem_gb: float,
    bytes_all_reduce_payload: int,
    bytes_all_reduce_theoretical: int,
    bytes_all_gather_payload: int,
    bytes_all_gather_theoretical: int,
) -> Dict[str, float]:
    row = {
        "run_kind": run_kind,
        "dataset": dataset,
        "world_size": world_size,
        "device": device,
        "dtype": dtype,
        "N_global": N_global,
        "d": d,
        "method": method,
        "covariance": covariance,
        "shrinkage": float(shrinkage),
        "reps": reps,
        "warmup": warmup,
        "peak_mem_gb": peak_mem_gb,
        "bytes_all_reduce_payload": bytes_all_reduce_payload,
        "bytes_all_reduce_theoretical": bytes_all_reduce_theoretical,
        "bytes_all_gather_payload": bytes_all_gather_payload,
        "bytes_all_gather_theoretical": bytes_all_gather_theoretical,
    }
    row.update(ms_stats)
    return row
