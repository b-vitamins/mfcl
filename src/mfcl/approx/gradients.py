from __future__ import annotations

from typing import Literal, Tuple

import math
import torch

from mfcl.approx.moments import compute_neg_stats, CovarianceKind, proj_mu, NegStats

ObjectiveKind = Literal["loob", "infonce"]


def _apply_cov(stats: NegStats, X: torch.Tensor) -> torch.Tensor:
    if stats.variant == "iso":
        assert stats.alpha is not None
        return stats.alpha * X
    if stats.variant == "diag":
        assert stats.diag is not None
        return X * stats.diag
    if stats.U is not None and stats.lam is not None and stats.alpha_tail is not None:
        proj = X @ stats.U
        head = (proj * stats.lam) @ stats.U.T
        return head + stats.alpha_tail * X
    assert stats.Sigma is not None
    return X @ stats.Sigma


def mf2_gradients(
    X: torch.Tensor,
    Y: torch.Tensor,
    tau: float,
    *,
    objective: ObjectiveKind = "infonce",
    covariance: CovarianceKind = "diag",
    shrinkage: float = 0.0,
    ridge: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    beta = 1.0 / float(tau)
    N, d = X.shape
    assert Y.shape == (N, d)

    gX = -(beta / N) * Y
    gY = -(beta / N) * X

    stats_y = compute_neg_stats(Y, variant=covariance, shrinkage=shrinkage, ridge=ridge)
    stats_x = compute_neg_stats(X, variant=covariance, shrinkage=shrinkage, ridge=ridge)

    cov_X = _apply_cov(stats_y, X)  # Σ_y x_i
    cov_Y = _apply_cov(stats_x, Y)  # Σ_x y_i

    if objective == "loob":
        dlogZx = beta * stats_y.mu[None, :] + (beta**2) * cov_X
        dlogZy = beta * stats_x.mu[None, :] + (beta**2) * cov_Y
        cross_x = torch.zeros_like(X)
        cross_y = torch.zeros_like(Y)
    else:
        m_x = (
            math.log(max(N - 1, 1))
            + beta * proj_mu(X, stats_y.mu)
            + 0.5 * (beta**2) * (cov_X * X).sum(dim=1)
        )
        a_x = beta * (X * Y).sum(dim=1)
        sig_x = torch.sigmoid(a_x - m_x)[:, None]
        dm_dx = beta * stats_y.mu[None, :] + (beta**2) * cov_X
        da_dx = beta * Y
        dlogZx = sig_x * da_dx + (1.0 - sig_x) * dm_dx

        m_y = (
            math.log(max(N - 1, 1))
            + beta * proj_mu(Y, stats_x.mu)
            + 0.5 * (beta**2) * (cov_Y * Y).sum(dim=1)
        )
        a_y = beta * (Y * X).sum(dim=1)
        sig_y = torch.sigmoid(a_y - m_y)[:, None]
        dm_dy = beta * stats_x.mu[None, :] + (beta**2) * cov_Y
        da_dy = beta * X
        dlogZy = sig_y * da_dy + (1.0 - sig_y) * dm_dy

        cross_x = sig_y * (beta * Y)
        cross_y = sig_x * (beta * X)

    gX = gX + (beta / (2 * N)) * (dlogZx + cross_x)
    gY = gY + (beta / (2 * N)) * (dlogZy + cross_y)
    return gX, gY
