"""SwAV swapped assignment loss with in-graph Sinkhorn-Knopp balancing."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVLoss(nn.Module):
    """SwAV swapped assignment with in-graph Sinkhorn."""

    def __init__(
        self,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        temperature: float = 0.1,
        use_float32_for_sinkhorn: bool = True,
        sinkhorn_tol: float = 1e-3,
        sinkhorn_max_iters: int = 100,
        force_fp32: bool = True,
        **_: dict,
    ) -> None:
        """Initialize SwAV loss.

        Args:
            epsilon: Entropic regularization coefficient (>0) for Sinkhorn.
            sinkhorn_iters: Number of row/col normalization iterations.
            temperature: Softmax temperature for prototype logits.
            use_float32_for_sinkhorn: Cast to float32 for Sinkhorn math for stability.

        Raises:
            ValueError: If epsilon <= 0, temperature <= 0, sinkhorn_iters < 1.
        """
        super().__init__()
        if epsilon <= 0 or temperature <= 0 or sinkhorn_iters < 1:
            raise ValueError(
                "Invalid SwAV parameters: ensure epsilon>0, temperature>0, iters>=1"
            )
        self.eps = float(epsilon)
        self.iters = int(sinkhorn_iters)
        self.temp = float(temperature)
        self.fp32_sinkhorn = bool(use_float32_for_sinkhorn)
        self.force_fp32 = bool(force_fp32)
        if sinkhorn_tol <= 0:
            raise ValueError("sinkhorn_tol must be > 0")
        if sinkhorn_max_iters < 1:
            raise ValueError("sinkhorn_max_iters must be >= 1")
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.sinkhorn_max_iters = int(sinkhorn_max_iters)

    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute balanced assignments via iterative Sinkhorn-Knopp with checks.

        Returns Q[B,K] with rows ~ 1 and cols ~ B/K.
        """
        B, K = scores.shape
        dtype = torch.float32 if self.fp32_sinkhorn else scores.dtype
        S = (scores.to(dtype) / self.eps)
        S = S - S.max(dim=1, keepdim=True).values
        Q = torch.exp(S)  # [B,K]

        a = torch.ones(B, 1, device=Q.device, dtype=Q.dtype)
        b = (B / K) * torch.ones(1, K, device=Q.device, dtype=Q.dtype)

        u = torch.ones(B, 1, device=Q.device, dtype=Q.dtype)
        v = torch.ones(1, K, device=Q.device, dtype=Q.dtype)
        eps = 1e-12
        tol = self.sinkhorn_tol
        it = 0
        max_iters = max(self.iters, self.sinkhorn_max_iters)
        while True:
            u = a / (Q @ v.t() + eps)
            v = b / (u.t() @ Q + eps)
            it += 1
            if it >= self.iters:
                P = (u * Q) * v
                if it >= max_iters:
                    Qb = P
                    break
                row_err = (P.sum(dim=1, keepdim=True) - a).abs().max().item()
                col_err = (P.sum(dim=0, keepdim=True) - b).abs().max().item()
                if max(row_err, col_err) <= tol:
                    Qb = P
                    break

        # Final tightening to ensure near-doubly-stochastic within tolerance
        rs = Qb.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Qb = Qb / rs
        cs = Qb.sum(dim=0, keepdim=True).clamp_min(1e-12)
        target_c = (B / K)
        Qb = Qb * (target_c / cs)
        rs2 = Qb.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Qb = Qb / rs2

        return Qb.detach()

    def forward(
        self,
        logits_per_crop: List[torch.Tensor],
        code_crop_indices: Tuple[int, int],
        queue_logits: Dict[int, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute SwAV loss.

        Args:
            logits_per_crop: List of length C where each element is [B, K] raw prototype
                logits for a crop (temperature is applied inside this loss).
            code_crop_indices: Indices of two "global" crops to compute assignments for,
                e.g. (0, 1).

        Returns:
            loss: Scalar swapped cross-entropy averaged over predicting all other crops.
            stats: {'entropy': mean code entropy, 'q_max_mean': mean max prob}
        """
        if not logits_per_crop:
            raise ValueError("logits_per_crop must be non-empty")
        C = len(logits_per_crop)
        B, K = logits_per_crop[0].shape
        for t in logits_per_crop:
            if t.ndim != 2 or t.shape != (B, K):
                raise ValueError("All crop logits must share shape [B,K]")
        a, b = code_crop_indices
        if not (0 <= a < C and 0 <= b < C):
            raise ValueError("code_crop_indices out of range")

        # Compute codes on the two designated crops
        queue_logits = queue_logits or {}

        def _codes_for(idx: int) -> torch.Tensor:
            base = logits_per_crop[idx]
            queued = queue_logits.get(idx)
            if queued is not None and queued.numel() > 0:
                combined = torch.cat([queued, base.detach()], dim=0)
                Q_full = self._sinkhorn(combined)
                return Q_full[-B:]
            return self._sinkhorn(base)

        Qa = _codes_for(a)
        Qb = _codes_for(b)

        # Stats from codes
        def entropy(Q: torch.Tensor) -> torch.Tensor:
            return -(Q * (Q.add(1e-12).log())).sum(dim=1).mean()

        q_stats_entropy = 0.5 * (entropy(Qa) + entropy(Qb))
        q_max_mean = 0.5 * (Qa.max(dim=1).values.mean() + Qb.max(dim=1).values.mean())

        # Predict assignments from other crops
        target_dtype = torch.float32 if self.force_fp32 else logits_per_crop[0].dtype
        logp_per_crop = [
            F.log_softmax(logits.to(target_dtype) / self.temp, dim=1)
            for logits in logits_per_crop
        ]
        losses = []
        for c, logits_c in enumerate(logits_per_crop):
            if c != a:
                logp = logp_per_crop[c]
                losses.append(-(Qa * logp).sum(dim=1).mean())
            if c != b:
                logp = logp_per_crop[c]
                losses.append(-(Qb * logp).sum(dim=1).mean())
        if not losses:
            raise ValueError("Not enough crops to compute SwAV loss")
        loss = torch.stack(losses, dim=0).mean()
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        stats = {"entropy": q_stats_entropy.detach(), "q_max_mean": q_max_mean.detach()}
        return loss, stats


__all__ = ["SwAVLoss"]
