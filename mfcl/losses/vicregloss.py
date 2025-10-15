"""VICReg loss: invariance + variance + covariance regularization."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _offdiag(M: torch.Tensor) -> torch.Tensor:
    n, m = M.shape
    assert n == m
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """VICReg: invariance + variance + covariance."""

    def __init__(
        self,
        lambda_invar: float = 25.0,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        """Initialize VICReg loss.

        Args:
            lambda_invar: Weight for invariance (MSE) term.
            mu_var: Weight for variance hinge term.
            nu_cov: Weight for covariance off-diagonal penalty.
            gamma: Target minimum std per dimension.
            eps: Numerical stability for std and centering.

        Raises:
            ValueError: If any weight <= 0 or gamma <= 0 or eps <= 0.
        """
        super().__init__()
        if lambda_invar <= 0 or mu_var <= 0 or gamma <= 0 or eps <= 0:
            raise ValueError("lambda_invar, mu_var must be > 0 and gamma, eps > 0")
        if nu_cov < 0:
            raise ValueError("nu_cov must be >= 0")
        self.lambda_invar = float(lambda_invar)
        self.mu_var = float(mu_var)
        self.nu_cov = float(nu_cov)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VICReg loss.

        Args:
            z1: [B, D] projections for view 1.
            z2: [B, D] projections for view 2.

        Returns:
            loss: Scalar.
            stats: {'mse': ..., 'std_mean': ..., 'cov_offdiag': ...}

        Raises:
            ValueError: If batch size < 2 or shapes mismatch.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("z1 and z2 must be 2D tensors with identical shapes")
        B = z1.shape[0]
        if B < 2:
            raise ValueError("batch size must be >= 2 for VICReg")

        x = z1.to(torch.float32)
        y = z2.to(torch.float32)

        # Invariance term (MSE between views)
        mse = F.mse_loss(x, y)

        # Variance term: hinge on per-dimension std
        sx = x.std(dim=0) + self.eps
        sy = y.std(dim=0) + self.eps
        vx = torch.relu(self.gamma - sx).mean()
        vy = torch.relu(self.gamma - sy).mean()
        var = 0.5 * (vx + vy)

        # Covariance term: off-diagonal penalty normalized by D
        xc = x - x.mean(dim=0)
        yc = y - y.mean(dim=0)
        Cx = (xc.t() @ xc) / (B - 1)
        Cy = (yc.t() @ yc) / (B - 1)
        cov = 0.5 * (
            ((_offdiag(Cx) ** 2).sum() / Cx.shape[0])
            + ((_offdiag(Cy) ** 2).sum() / Cy.shape[0])
        )

        loss = self.lambda_invar * mse + self.mu_var * var + self.nu_cov * cov
        stats = {
            "mse": mse.detach(),
            "std_mean": 0.5 * (sx.mean() + sy.mean()).detach(),
            "cov_offdiag": 0.5
            * (_offdiag(Cx).abs().mean() + _offdiag(Cy).abs().mean()).detach(),
        }
        return loss, stats


__all__ = ["VICRegLoss"]
