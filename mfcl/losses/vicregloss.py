"""VICReg loss: invariance + variance + covariance regularization."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _offdiag(M: torch.Tensor) -> torch.Tensor:
    n, m = M.shape
    if n != m:
        raise ValueError("covariance matrix must be square to extract off-diagonal entries")
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """VICReg: invariance + variance + covariance."""

    def __init__(
        self,
        lambda_inv: float | None = None,
        lambda_invar: float | None = None,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        force_fp32: bool = True,
    ) -> None:
        """Initialize VICReg loss.

        Args:
            lambda_inv: Weight for invariance (MSE) term. Preferred kw name.
            lambda_invar: Deprecated alias for :param:`lambda_inv`.
            mu_var: Weight for variance hinge term.
            nu_cov: Weight for covariance off-diagonal penalty.
            gamma: Target minimum std per dimension.
            eps: Numerical stability for std and centering.

        Raises:
            ValueError: If both lambda arguments provided or weights invalid.
        """
        super().__init__()

        if lambda_inv is not None and lambda_invar is not None:
            raise ValueError("Specify only one of lambda_inv or lambda_invar")
        if lambda_inv is None and lambda_invar is None:
            lambda_inv = 25.0
        elif lambda_inv is None:
            lambda_inv = lambda_invar

        if lambda_inv is None:
            raise ValueError("lambda_inv must be provided")

        if lambda_inv <= 0 or mu_var <= 0 or gamma <= 0 or eps <= 0:
            raise ValueError("lambda_inv, mu_var must be > 0 and gamma, eps > 0")
        if nu_cov < 0:
            raise ValueError("nu_cov must be >= 0")

        self.lambda_inv = float(lambda_inv)
        self.mu_var = float(mu_var)
        self.nu_cov = float(nu_cov)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.force_fp32 = bool(force_fp32)

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

        Notes:
            Standard deviations are computed with unbiased=False for stability.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("z1 and z2 must be 2D tensors with identical shapes")
        B, _ = z1.shape
        if B < 2:
            raise ValueError("batch size must be >= 2 for VICReg")

        target_dtype = torch.float32 if self.force_fp32 else z1.dtype
        x = z1.to(target_dtype)
        y = z2.to(target_dtype)

        # Invariance term (MSE between views)
        mse = F.mse_loss(x, y)

        # Variance term: hinge on per-dimension std
        sx = x.std(dim=0, unbiased=False) + self.eps
        sy = y.std(dim=0, unbiased=False) + self.eps
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

        loss = self.lambda_inv * mse + self.mu_var * var + self.nu_cov * cov
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)
        stats = {
            "mse": mse.detach(),
            "std_mean": 0.5 * (sx.mean() + sy.mean()).detach(),
            "cov_offdiag": 0.5
            * (_offdiag(Cx).abs().mean() + _offdiag(Cy).abs().mean()).detach(),
        }
        return loss, stats


__all__ = ["VICRegLoss"]
