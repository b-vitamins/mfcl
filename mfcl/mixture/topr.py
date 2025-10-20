"""Top-R diagnostics for mixture responsibilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class TopRSelection:
    """Result of selecting the top-R mixture components per anchor."""

    indices: torch.Tensor
    weights: torch.Tensor
    epsilon: torch.Tensor
    log_norm: torch.Tensor


def _normalize_pi(pi: torch.Tensor, pi_floor: float) -> torch.Tensor:
    if pi.dim() != 1:
        raise ValueError("pi must be a 1D tensor")
    pi = pi.to(dtype=torch.float32)
    pi = torch.clamp(pi, min=float(pi_floor))
    total = pi.sum()
    if total <= 0:
        raise ValueError("pi must have positive mass")
    return pi / total


def select_topR(
    Q: torch.Tensor,
    pi: torch.Tensor,
    R: int,
    *,
    pi_floor: float = 1e-6,
    chunk_size: Optional[int] = None,
) -> TopRSelection:
    """Select the Top-R mixture components per anchor."""

    if Q.dim() != 2:
        raise ValueError("Q must be a [N, K] tensor")
    if R < 0:
        raise ValueError("R must be non-negative")

    n, k = Q.shape
    if pi.numel() != k:
        raise ValueError("pi must have K entries matching Q.shape[1]")

    pi_norm = _normalize_pi(pi, pi_floor)
    log_pi = torch.log(pi_norm.to(device=Q.device, dtype=Q.dtype))

    top_k = k if R == 0 else min(R, k)

    if chunk_size is None or chunk_size <= 0 or chunk_size >= k:
        logits = Q + log_pi
        log_norm = torch.logsumexp(logits, dim=1)
        if top_k == 0:
            top_idx = torch.empty(n, 0, dtype=torch.long, device=Q.device)
            top_weights = torch.empty(n, 0, dtype=Q.dtype, device=Q.device)
            epsilon = torch.ones(n, dtype=Q.dtype, device=Q.device)
            return TopRSelection(indices=top_idx, weights=top_weights, epsilon=epsilon, log_norm=log_norm)

        top_logits, top_idx = torch.topk(logits, k=top_k, dim=1)
        weights = torch.exp(top_logits - log_norm.unsqueeze(1))
        epsilon = torch.clamp(1.0 - weights.sum(dim=1), min=0.0, max=1.0)
        top_weights = weights
        return TopRSelection(indices=top_idx, weights=top_weights, epsilon=epsilon, log_norm=log_norm)

    chunk = int(chunk_size)
    max_logits = torch.full((n,), float("-inf"), dtype=Q.dtype, device=Q.device)
    for start in range(0, k, chunk):
        end = min(k, start + chunk)
        block = Q[:, start:end] + log_pi[start:end]
        block_max = block.max(dim=1).values
        max_logits = torch.maximum(max_logits, block_max)

    sum_exp = torch.zeros(n, dtype=Q.dtype, device=Q.device)
    for start in range(0, k, chunk):
        end = min(k, start + chunk)
        block = Q[:, start:end] + log_pi[start:end]
        sum_exp += torch.exp(block - max_logits.unsqueeze(1)).sum(dim=1)
    sum_exp = torch.clamp(sum_exp, min=torch.finfo(Q.dtype).tiny)
    log_norm = max_logits + torch.log(sum_exp)

    if top_k == 0:
        top_idx = torch.empty(n, 0, dtype=torch.long, device=Q.device)
        top_weights = torch.empty(n, 0, dtype=Q.dtype, device=Q.device)
        epsilon = torch.ones(n, dtype=Q.dtype, device=Q.device)
        return TopRSelection(indices=top_idx, weights=top_weights, epsilon=epsilon, log_norm=log_norm)

    top_logits = torch.full((n, top_k), float("-inf"), dtype=Q.dtype, device=Q.device)
    top_indices = torch.full((n, top_k), -1, dtype=torch.long, device=Q.device)
    for start in range(0, k, chunk):
        end = min(k, start + chunk)
        block = Q[:, start:end] + log_pi[start:end]
        block_idx = torch.arange(start, end, device=Q.device)
        idx_block = block_idx.unsqueeze(0).expand(n, -1)
        combined_logits = torch.cat([top_logits, block], dim=1)
        combined_indices = torch.cat([top_indices, idx_block], dim=1)
        new_top_logits, order = torch.topk(combined_logits, k=top_k, dim=1)
        top_logits = new_top_logits
        top_indices = torch.gather(combined_indices, 1, order)

    top_weights = torch.exp(top_logits - log_norm.unsqueeze(1))
    epsilon = torch.clamp(1.0 - top_weights.sum(dim=1), min=0.0, max=1.0)
    return TopRSelection(indices=top_indices, weights=top_weights, epsilon=epsilon, log_norm=log_norm)


def _pairwise_max_mu(mu: torch.Tensor) -> float:
    if mu.dim() != 2:
        raise ValueError("mu must be [K, D]")
    k = mu.shape[0]
    if k <= 1:
        return 0.0
    max_val = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
    for idx in range(k - 1):
        diffs = mu[idx + 1 :] - mu[idx]
        if diffs.numel() == 0:
            continue
        norms = torch.linalg.norm(diffs, dim=1)
        max_val = torch.maximum(max_val, norms.max())
    return float(max_val.detach().cpu().item())


def _pairwise_max_sigma(sigma: torch.Tensor) -> float:
    if sigma.dim() != 3:
        raise ValueError("Sigma must be [K, D, D]")
    k = sigma.shape[0]
    if k <= 1:
        return 0.0
    max_val = torch.tensor(0.0, dtype=sigma.dtype, device=sigma.device)
    for idx in range(k - 1):
        diffs = sigma[idx + 1 :] - sigma[idx]
        if diffs.numel() == 0:
            continue
        norms = torch.linalg.matrix_norm(diffs, ord=2)
        max_val = torch.maximum(max_val, norms.max())
    return float(max_val.detach().cpu().item())


def _quantile(tensor: torch.Tensor, q: float) -> float:
    if tensor.numel() == 0:
        return float("nan")
    return float(torch.quantile(tensor, q).item())


class TopRDiagnostics:
    """Track Top-R responsibility diagnostics and log quantiles."""

    _CSV_COLUMNS = ["step", "epoch", "R", "epsilon_p50", "epsilon_p90", "err_bound_p50", "err_bound_p90"]

    def __init__(
        self,
        *,
        R: int,
        enabled: bool,
        log_dir: Optional[str | Path] = None,
        is_main: bool = True,
        pi_floor: float = 1e-6,
        chunk_size: Optional[int] = None,
    ) -> None:
        if R < 0:
            raise ValueError("R must be non-negative")
        self.R = int(R)
        self.enabled = bool(enabled)
        self.pi_floor = float(pi_floor)
        self.chunk_size = int(chunk_size) if chunk_size is not None and chunk_size > 0 else None

        self._last_metrics: Dict[str, torch.Tensor | float] = {}
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        if log_dir is not None and self.enabled and is_main:
            path = Path(log_dir).joinpath("topr.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_COLUMNS)
            if path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._csv_file.flush()

    @property
    def last_metrics(self) -> Dict[str, torch.Tensor | float]:
        return dict(self._last_metrics)

    def close(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._csv_writer = None

    def update(
        self,
        *,
        responsibilities: Optional[torch.Tensor],
        pi: Optional[torch.Tensor],
        mu: Optional[torch.Tensor],
        Sigma: Optional[torch.Tensor],
        beta: Optional[float | torch.Tensor],
        Q: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if not self.enabled:
            return {}
        if pi is None or mu is None or Sigma is None:
            return {}

        with torch.no_grad():
            pi_t = pi.detach().to(torch.float32)
            mu_t = mu.detach().to(torch.float32)
            sigma_t = Sigma.detach().to(torch.float32)

            eps_values: Optional[torch.Tensor] = None
            if responsibilities is not None:
                resp = responsibilities.detach().to(torch.float32)
                if resp.ndim != 2:
                    raise ValueError("responsibilities must be [N, K]")
                top_k = resp.shape[1] if self.R == 0 else min(self.R, resp.shape[1])
                if top_k == 0:
                    eps_values = torch.ones(resp.shape[0], dtype=resp.dtype, device=resp.device)
                else:
                    top_vals, _ = torch.topk(resp, k=top_k, dim=1)
                    eps_values = torch.clamp(1.0 - top_vals.sum(dim=1), min=0.0, max=1.0)
            elif Q is not None:
                selection = select_topR(
                    Q=Q.detach().to(torch.float32),
                    pi=pi_t,
                    R=self.R,
                    pi_floor=self.pi_floor,
                    chunk_size=self.chunk_size,
                )
                eps_values = selection.epsilon.detach()
            else:
                return {}

            if eps_values.numel() == 0:
                return {}

            eps_cpu = eps_values.to(torch.float32).cpu()
            D_mu = _pairwise_max_mu(mu_t)
            D_sigma = _pairwise_max_sigma(sigma_t)
            beta_val = 0.0
            if beta is not None:
                try:
                    beta_val = float(beta)
                except (TypeError, ValueError):
                    beta_val = 0.0
            abs_beta = abs(beta_val)
            scale = abs_beta * D_mu + (abs_beta ** 2) * D_sigma
            if scale == 0.0:
                err = torch.zeros_like(eps_cpu)
            else:
                denom = torch.clamp(1.0 - eps_cpu, min=1e-8)
                err = eps_cpu / denom * scale
                inf_mask = eps_cpu >= (1.0 - 1e-8)
                if inf_mask.any():
                    err[inf_mask] = float("inf")

            metrics: Dict[str, torch.Tensor | float] = {
                "epsilon": eps_cpu,
                "err_bound": err,
                "epsilon_p50": _quantile(eps_cpu, 0.5),
                "epsilon_p90": _quantile(eps_cpu, 0.9),
                "err_bound_p50": _quantile(err, 0.5),
                "err_bound_p90": _quantile(err, 0.9),
            }
            self._last_metrics = metrics
            return dict(metrics)

    def log_step(self, *, step: int, epoch: int) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return
        if not self._last_metrics:
            return
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "R": int(self.R),
            "epsilon_p50": self._last_metrics.get("epsilon_p50"),
            "epsilon_p90": self._last_metrics.get("epsilon_p90"),
            "err_bound_p50": self._last_metrics.get("err_bound_p50"),
            "err_bound_p90": self._last_metrics.get("err_bound_p90"),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()


__all__ = ["TopRSelection", "TopRDiagnostics", "select_topR"]
