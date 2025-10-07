from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


CovarianceKind = Literal["iso", "diag", "full"]


@dataclass
class NegStats:
    """Sufficient statistics of the negatives for MF2."""

    variant: CovarianceKind
    mu: torch.Tensor  # [d]
    alpha: Optional[float] = None  # iso variance scalar
    diag: Optional[torch.Tensor] = None  # [d] for diag
    Sigma: Optional[torch.Tensor] = None  # [d, d] for full
    U: Optional[torch.Tensor] = None  # [d, r] low-rank head
    lam: Optional[torch.Tensor] = None  # [r]
    alpha_tail: Optional[float] = None  # scalar tail mass per dim
    ridge: float = 1e-5

    @torch.no_grad()
    def quad(self, X: torch.Tensor) -> torch.Tensor:
        """
        Rowwise quadratic form x_i^T Σ x_i based on the chosen variant.
        Returns [N].
        """
        if self.variant == "iso":
            assert self.alpha is not None
            return self.alpha * (X * X).sum(dim=1)

        if self.variant == "diag":
            assert self.diag is not None
            return (X * X) @ self.diag

        if self.variant == "full":
            if (
                self.U is not None
                and self.lam is not None
                and self.alpha_tail is not None
            ):
                proj = X @ self.U  # [N, r]
                head = (proj * proj) @ self.lam  # [N]
                tail = self.alpha_tail * (X * X).sum(dim=1)
                return head + tail
            assert self.Sigma is not None
            XS = X @ self.Sigma
            return (XS * X).sum(dim=1)

        raise ValueError(f"Unknown variant: {self.variant}")


@torch.no_grad()
def _center(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = Y.mean(dim=0)
    Yc = Y - mu
    return mu, Yc


@torch.no_grad()
def _cov_full(Yc: torch.Tensor) -> torch.Tensor:
    N = Yc.shape[0]
    return (Yc.T @ Yc) / float(N)


@torch.no_grad()
def _cov_diag(Yc: torch.Tensor) -> torch.Tensor:
    return (Yc * Yc).mean(dim=0)


@torch.no_grad()
def compute_neg_stats(
    Y: torch.Tensor,
    variant: CovarianceKind = "diag",
    *,
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
    lowrank_r: Optional[int] = None,
) -> NegStats:
    """
    Compute μ and Σ summary for negatives according to variant:
      - iso: Σ = α I with α = mean(diag(Σ_full))
      - diag: Σ = diag(v) with v = per-dimension variance
      - full: Σ = (1-λ) Σ_hat + λ * (tr(Σ_hat)/d) I, then add small ridge
              If lowrank_r is provided, Sigma is represented as U diag(lam) U^T + α_tail I.
    """
    assert 0.0 <= shrinkage <= 1.0, "shrinkage must be in [0,1]"
    mu, Yc = _center(Y)
    d = Y.shape[1]

    if variant == "iso":
        diag = _cov_diag(Yc)
        alpha = float(diag.mean().item()) + ridge
        return NegStats(variant="iso", mu=mu, alpha=alpha, ridge=ridge)

    if variant == "diag":
        diag = _cov_diag(Yc) + ridge
        return NegStats(variant="diag", mu=mu, diag=diag, ridge=ridge)

    if variant == "full":
        Sigma = _cov_full(Yc)
        tr_over_d = float(torch.trace(Sigma).item()) / float(d)
        if shrinkage > 0.0:
            Sigma = (1.0 - shrinkage) * Sigma + shrinkage * tr_over_d * torch.eye(
                d, device=Y.device, dtype=Y.dtype
            )
        if ridge > 0.0:
            Sigma = Sigma + ridge * torch.eye(d, device=Y.device, dtype=Y.dtype)

        if lowrank_r is not None and 0 < lowrank_r < d:
            evals, evecs = torch.linalg.eigh(Sigma)
            evals = evals.flip(0)
            evecs = evecs.flip(1)
            r = int(lowrank_r)
            lam_head = evals[:r].clamp_min(0)
            U = evecs[:, :r]
            tail_sum = torch.clamp(evals[r:], min=0).sum().item()
            alpha_tail = float(tail_sum / d)
            return NegStats(
                variant="full",
                mu=mu,
                U=U,
                lam=lam_head,
                alpha_tail=alpha_tail,
                ridge=ridge,
            )

        return NegStats(variant="full", mu=mu, Sigma=Sigma, ridge=ridge)

    raise ValueError(f"Unknown covariance variant: {variant}")


@torch.no_grad()
def ema_update_diag(
    mu_prev: torch.Tensor,
    var_prev: torch.Tensor,
    Y: torch.Tensor,
    decay: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_batch, Yc = _center(Y)
    var_batch = _cov_diag(Yc)
    mu_new = decay * mu_prev + (1 - decay) * mu_batch
    var_new = decay * var_prev + (1 - decay) * var_batch
    return mu_new, var_new


@torch.no_grad()
def negstats_from_diag(
    mu: torch.Tensor, var: torch.Tensor, use_iso: bool = False, ridge: float = 1e-5
) -> NegStats:
    if use_iso:
        alpha = float(var.mean().item()) + ridge
        return NegStats(variant="iso", mu=mu, alpha=alpha, ridge=ridge)
    return NegStats(variant="diag", mu=mu, diag=var + ridge, ridge=ridge)


@torch.no_grad()
def proj_mu(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Compute x_i^T mu for all rows."""
    return X @ mu
