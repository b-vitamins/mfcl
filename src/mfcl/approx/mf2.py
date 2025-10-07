from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from mfcl.approx.moments import compute_neg_stats, NegStats, proj_mu, CovarianceKind


@torch.no_grad()
def mf2_loob_centered(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    stats: Optional[NegStats] = None,
    covariance: CovarianceKind = "diag",
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
) -> torch.Tensor:
    """
    MF2 approximation of centered LOOB log-mass:
      \tilde{ell}_i ≈ β x_i^T μ_y + (β^2/2) x_i^T Σ_y x_i
    """
    beta = 1.0 / float(tau)
    if stats is None:
        stats = compute_neg_stats(
            Y, variant=covariance, shrinkage=shrinkage, ridge=ridge
        )
    mu_proj = proj_mu(X, stats.mu)  # [N]
    quad = stats.quad(X)  # [N]
    return beta * mu_proj + 0.5 * (beta**2) * quad  # [N]


@torch.no_grad()
def mf2_infonce_logmass(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    stats: Optional[NegStats] = None,
    covariance: CovarianceKind = "diag",
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
) -> torch.Tensor:
    """
    MF2 approximation of InfoNCE log normalization per anchor:
      log Z_i ≈ m_i + log(1 + exp(a_i - m_i))
      where m_i = log(N-1) + β x_i^T μ_y + (β^2/2) x_i^T Σ_y x_i
            a_i = β x_i^T y_i
    """
    beta = 1.0 / float(tau)
    N = X.shape[0]
    if stats is None:
        stats = compute_neg_stats(
            Y, variant=covariance, shrinkage=shrinkage, ridge=ridge
        )

    mu_proj = proj_mu(X, stats.mu)  # [N]
    quad = stats.quad(X)  # [N]
    m_neg = math.log(max(N - 1, 1)) + beta * mu_proj + 0.5 * (beta**2) * quad  # [N]
    a_pos = beta * (X * Y).sum(dim=1)  # [N]

    # logZ = m_neg + log(1 + exp(a_pos - m_neg)) with numerically stable softplus
    return m_neg + F.softplus(a_pos - m_neg)
