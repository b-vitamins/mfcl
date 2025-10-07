from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

import torch


@dataclass
class SummaryStats:
    mean: float
    std: float
    min: float
    max: float
    q25: float
    median: float
    q75: float


@torch.no_grad()
def summarize_vector(v: torch.Tensor) -> SummaryStats:
    v = v.detach().to("cpu")
    return SummaryStats(
        mean=float(v.mean().item()),
        std=float(v.std(unbiased=False).item()),
        min=float(v.min().item()),
        max=float(v.max().item()),
        q25=float(v.quantile(0.25).item()),
        median=float(v.quantile(0.5).item()),
        q75=float(v.quantile(0.75).item()),
    )


@torch.no_grad()
def normalization_summary_row(
    values: torch.Tensor,
    *,
    dataset: str,
    objective: str,
    N: int,
    tau: float,
    batch_id: int,
    method: str = "exact",
) -> Dict[str, Union[str, float, int]]:
    s = summarize_vector(values)
    return {
        "dataset": dataset,
        "objective": objective,
        "method": method,
        "N": N,
        "tau": tau,
        "batch_id": batch_id,
        "mean": s.mean,
        "std": s.std,
        "q25": s.q25,
        "median": s.median,
        "q75": s.q75,
        "min": s.min,
        "max": s.max,
    }


@torch.no_grad()
def error_metrics(
    approx: torch.Tensor,
    exact: torch.Tensor,
) -> Dict[str, float]:
    a = approx.detach().to("cpu")
    e = exact.detach().to("cpu")
    diff = a - e
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    bias = float(diff.mean().item())
    std_exact = float(e.std(unbiased=False).item())
    rel_mae = float(mae / std_exact) if std_exact > 0 else 0.0
    viol_lt = float((diff < 0).float().mean().item())
    viol_gt = float((diff > 0).float().mean().item())
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "std_exact": std_exact,
        "rel_mae": rel_mae,
        "viol_lt": viol_lt,
        "viol_gt": viol_gt,
    }


@torch.no_grad()
def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.to("cpu")
    y = y.to("cpu")
    vx = x - x.mean()
    vy = y - y.mean()
    sx = vx.std(unbiased=False)
    sy = vy.std(unbiased=False)
    if sx.item() == 0 or sy.item() == 0:
        return float("nan")
    return float((vx * vy).mean().item() / (sx.item() * sy.item()))


@torch.no_grad()
def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    # rank via argsort twice
    def rank(t: torch.Tensor) -> torch.Tensor:
        idx = torch.argsort(t)
        r = torch.empty_like(idx, dtype=torch.float32)
        r[idx] = torch.arange(1, t.numel() + 1, dtype=torch.float32)
        return r

    return _pearsonr(rank(x), rank(y))


def default_tost_epsilon(tau: float) -> float:
    if tau <= 0.05 + 1e-9:
        return 0.04
    if tau <= 0.10 + 1e-9:
        return 0.02
    return 0.01


@torch.no_grad()
def bootstrap_mae_ci(
    approx: torch.Tensor,
    exact: torch.Tensor,
    *,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 1234,
) -> Tuple[float, float]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = approx.detach().to("cpu")
    e = exact.detach().to("cpu")
    N = a.numel()
    if N == 0:
        return float("nan"), float("nan")
    mae_samples = []
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=gen)
        d = (a[idx] - e[idx]).abs().mean().item()
        mae_samples.append(d)
    vals = torch.tensor(mae_samples)
    low = float(vals.quantile(alpha / 2).item())
    high = float(vals.quantile(1 - alpha / 2).item())
    return low, high


@torch.no_grad()
def error_summary_row(
    approx: torch.Tensor,
    exact: torch.Tensor,
    *,
    dataset: str,
    objective: str,
    method: str,
    covariance: str,
    shrinkage: float,
    N: int,
    tau: float,
    batch_id: int,
    tost_eps: float,
    n_boot: int = 200,
    alpha: float = 0.10,
    gap: Optional[torch.Tensor] = None,
    extra: Optional[Dict[str, Union[str, float, int]]] = None,
) -> Dict[str, Union[str, float, int]]:
    em = error_metrics(approx, exact)
    ci_low, ci_high = bootstrap_mae_ci(approx, exact, n_boot=n_boot, alpha=alpha)
    tost_pass = float(ci_high <= tost_eps)
    row = {
        "dataset": dataset,
        "objective": objective,
        "method": method,
        "covariance": covariance,
        "shrinkage": float(shrinkage),
        "N": N,
        "tau": float(tau),
        "batch_id": batch_id,
        "mae": em["mae"],
        "rmse": em["rmse"],
        "bias": em["bias"],
        "std_exact": em["std_exact"],
        "rel_mae": em["rel_mae"],
        "viol_lt": em["viol_lt"],
        "viol_gt": em["viol_gt"],
        "tost_eps": float(tost_eps),
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
        "tost_pass": tost_pass,
        "n_boot": n_boot,
        "alpha": alpha,
    }
    if gap is not None:
        diff = (approx - exact).detach()
        row["gap_mean"] = float(gap.mean().item())
        row["gap_q75"] = float(gap.quantile(0.75).item())
        row["corr_pearson"] = _pearsonr(diff, gap)
        row["corr_spearman"] = _spearmanr(diff, gap)
    if extra:
        row.update(extra)
    return row
