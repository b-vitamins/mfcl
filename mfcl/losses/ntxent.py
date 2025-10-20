"""NT-Xent (SimCLR) loss.

Supports both the standard "paired" 2-view variant (B−1 negatives per anchor)
and the original 2N formulation contrasting across both views (2B−2 negatives).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfcl.utils import dist as dist_utils
from mfcl.telemetry.hardness import get_active_monitor
from mfcl.losses.base import SelfSupervisedLoss

class NTXentLoss(SelfSupervisedLoss):
    """Normalized temperature-scaled cross-entropy for two views.

    Returns:
        loss: Scalar tensor, mean over the batch. The loss uses mean reduction.

    Notes:
        Similarities are computed in float32 for stability.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
        mode: str = "paired",
        cross_rank_negatives: bool = False,
        force_fp32: bool = True,
    ) -> None:
        """Initialize NT-Xent loss.

        Args:
            temperature: Softmax temperature τ > 0.
            normalize: If True, L2-normalize embeddings before similarity.
            mode: "paired" for the efficient 2-view loss or "2n"/"twoN" for
                the 2B×2B variant with 2B−2 negatives per anchor.

        Raises:
            ValueError: If temperature <= 0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.t = float(temperature)
        self.normalize = bool(normalize)
        mode_lc = mode.lower()
        if mode_lc in {"paired"}:
            self.mode = "paired"
        elif mode_lc in {"2n", "twon"}:
            self.mode = "2n"
        else:
            raise ValueError("mode must be 'paired' or '2N'")
        self.cross_rank_negatives = bool(cross_rank_negatives)
        self.force_fp32 = bool(force_fp32)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute symmetric NT-Xent.

        Args:
            z1: [B, D] projections for view 1.
            z2: [B, D] projections for view 2.

        Returns:
            loss: Scalar tensor, mean over the batch. The loss uses mean reduction.
            stats: {'pos_sim': mean cosine of positives,
                    'neg_sim_mean': mean cosine over negatives}

        Raises:
            ValueError: If batch size < 2 or shapes mismatch.

        Notes:
            Similarities are computed in float32 for stability.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("z1 and z2 must be 2D tensors with identical shapes")
        B = z1.shape[0]
        if B < 2:
            raise ValueError("batch size must be >= 2 for NT-Xent")

        target_dtype = torch.float32 if self.force_fp32 else z1.dtype
        z1f = z1.to(target_dtype)
        z2f = z2.to(target_dtype)
        if self.normalize:
            z1f = F.normalize(z1f, dim=1)
            z2f = F.normalize(z2f, dim=1)

        if self.mode == "paired":
            return self._paired(z1f, z2f)
        return self._two_n(z1f, z2f)

    def _paired(
        self, z1f: torch.Tensor, z2f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cross_rank_negatives and dist_utils.get_world_size() > 1:
            return self._paired_cross_rank(z1f, z2f)
        sim = z1f @ z2f.t()
        logits_i = sim / self.t
        logits_j = logits_i.t()
        labels = torch.arange(z1f.shape[0], device=sim.device)

        loss_i = F.cross_entropy(logits_i, labels)
        loss_j = F.cross_entropy(logits_j, labels)
        loss = 0.5 * (loss_i + loss_j)
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        pos_sim = torch.diag(sim).mean().detach()
        neg_mask = ~torch.eye(z1f.shape[0], dtype=torch.bool, device=sim.device)
        monitor = get_active_monitor()
        if monitor is not None:
            negatives = sim.detach().to(torch.float32).masked_select(neg_mask).view(z1f.shape[0], -1)
            monitor.add_negatives(negatives)
        neg_sim_mean = sim.masked_select(neg_mask).mean().detach()
        stats = {"pos_sim": pos_sim, "neg_sim_mean": neg_sim_mean}
        return loss, stats

    def _paired_cross_rank(
        self, z1f: torch.Tensor, z2f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = z1f.shape[0]
        rank = dist_utils.get_rank()

        def _gather_preserve_local(tensor: torch.Tensor) -> torch.Tensor:
            gathered = dist_utils.all_gather_tensor(tensor)
            world = dist_utils.get_world_size()
            if world <= 1:
                return gathered
            offset = rank * B
            pieces = []
            if offset > 0:
                pieces.append(gathered[:offset].detach())
            pieces.append(tensor)
            end = offset + B
            if end < gathered.shape[0]:
                pieces.append(gathered[end:].detach())
            if len(pieces) == 1:
                return pieces[0]
            return torch.cat(pieces, dim=0)

        z1_all = _gather_preserve_local(z1f)
        z2_all = _gather_preserve_local(z2f)
        offset = rank * B

        sim_i = z1f @ z2_all.t()
        logits_i = sim_i / self.t
        targets = torch.arange(B, device=z1f.device) + offset
        loss_i = F.cross_entropy(logits_i, targets)

        sim_j = z2f @ z1_all.t()
        logits_j = sim_j / self.t
        loss_j = F.cross_entropy(logits_j, targets)

        loss = 0.5 * (loss_i + loss_j)
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        pos_sim = torch.sum(z1f * z2f, dim=1).mean().detach()
        mask_i = torch.ones_like(sim_i, dtype=torch.bool)
        mask_i[torch.arange(B, device=sim_i.device), targets] = False
        mask_j = torch.ones_like(sim_j, dtype=torch.bool)
        mask_j[torch.arange(B, device=sim_j.device), targets] = False
        neg_vals = []
        if mask_i.any():
            neg_vals.append(sim_i.masked_select(mask_i))
        if mask_j.any():
            neg_vals.append(sim_j.masked_select(mask_j))
        monitor = get_active_monitor()
        if monitor is not None:
            if mask_i.any():
                vals_i = sim_i.detach().to(torch.float32).masked_select(mask_i).view(B, -1)
                monitor.add_negatives(vals_i)
            if mask_j.any():
                vals_j = sim_j.detach().to(torch.float32).masked_select(mask_j).view(B, -1)
                monitor.add_negatives(vals_j)
        if neg_vals:
            neg_sim_mean = torch.cat(neg_vals).mean().detach()
        else:
            neg_sim_mean = sim_i.new_zeros(())
        stats = {"pos_sim": pos_sim, "neg_sim_mean": neg_sim_mean}
        return loss, stats

    def _two_n(
        self, z1f: torch.Tensor, z2f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cross_rank_negatives and dist_utils.get_world_size() > 1:
            return self._two_n_cross_rank(z1f, z2f)
        z_all = torch.cat([z1f, z2f], dim=0)
        sim_full = z_all @ z_all.t()
        logits = sim_full / self.t
        B = z1f.shape[0]
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(eye, neg_inf)
        labels = torch.arange(B, device=logits.device)
        targets = torch.cat([labels + B, labels], dim=0)
        loss = F.cross_entropy(logits, targets)
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        diag_pos = torch.diag(sim_full, diagonal=B)
        diag_pos_rev = torch.diag(sim_full, diagonal=-B)
        pos_entries = torch.cat([diag_pos, diag_pos_rev])
        pos_sim = pos_entries.mean().detach()
        pos_mask = torch.zeros_like(sim_full, dtype=torch.bool)
        pos_mask[:B, B:] = torch.eye(B, dtype=torch.bool, device=sim_full.device)
        pos_mask[B:, :B] = torch.eye(B, dtype=torch.bool, device=sim_full.device)
        neg_mask = (~eye) & (~pos_mask)
        monitor = get_active_monitor()
        if monitor is not None:
            neg_values = sim_full.detach().to(torch.float32).masked_select(neg_mask).view(sim_full.shape[0], -1)
            monitor.add_negatives(neg_values)
        neg_sim_mean = sim_full.masked_select(neg_mask).mean().detach()
        stats = {"pos_sim": pos_sim, "neg_sim_mean": neg_sim_mean}
        return loss, stats

    def _two_n_cross_rank(
        self, z1f: torch.Tensor, z2f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = z1f.shape[0]
        rank = dist_utils.get_rank()

        def _gather_preserve_local(tensor: torch.Tensor) -> torch.Tensor:
            gathered = dist_utils.all_gather_tensor(tensor)
            world = dist_utils.get_world_size()
            if world <= 1:
                return gathered
            offset = rank * B
            pieces = []
            if offset > 0:
                pieces.append(gathered[:offset].detach())
            pieces.append(tensor)
            end = offset + B
            if end < gathered.shape[0]:
                pieces.append(gathered[end:].detach())
            if len(pieces) == 1:
                return pieces[0]
            return torch.cat(pieces, dim=0)

        z1_all = _gather_preserve_local(z1f)
        z2_all = _gather_preserve_local(z2f)
        world_B = z1_all.shape[0]
        offset = rank * B

        anchors = torch.cat([z1f, z2f], dim=0)
        all_feats = torch.cat([z1_all, z2_all], dim=0)
        logits = anchors @ all_feats.t() / self.t
        neg_inf = torch.finfo(logits.dtype).min

        mask = torch.zeros_like(logits, dtype=torch.bool)
        idx = torch.arange(B, device=logits.device)
        mask[idx, offset + idx] = True
        mask[idx + B, world_B + offset + idx] = True
        logits = logits.masked_fill(mask, neg_inf)

        targets = torch.cat(
            [offset + idx + world_B, offset + idx], dim=0
        )
        loss = F.cross_entropy(logits, targets)
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)
        pos_vals = torch.cat(
            [torch.sum(z1f * z2f, dim=1), torch.sum(z2f * z1f, dim=1)]
        )
        pos_sim = pos_vals.mean().detach()

        sim_i = z1f @ z2_all.t()
        sim_j = z2f @ z1_all.t()
        mask_i = torch.ones_like(sim_i, dtype=torch.bool)
        mask_i[idx, offset + idx] = False
        mask_j = torch.ones_like(sim_j, dtype=torch.bool)
        mask_j[idx, offset + idx] = False
        neg_parts = []
        if mask_i.any():
            neg_parts.append(sim_i.masked_select(mask_i))
        if mask_j.any():
            neg_parts.append(sim_j.masked_select(mask_j))
        monitor = get_active_monitor()
        if monitor is not None:
            if mask_i.any():
                vals_i = sim_i.detach().to(torch.float32).masked_select(mask_i).view(B, -1)
                monitor.add_negatives(vals_i)
            if mask_j.any():
                vals_j = sim_j.detach().to(torch.float32).masked_select(mask_j).view(B, -1)
                monitor.add_negatives(vals_j)
        if neg_parts:
            neg_sim_mean = torch.cat(neg_parts).mean().detach()
        else:
            neg_sim_mean = logits.new_zeros(())

        stats = {"pos_sim": pos_sim, "neg_sim_mean": neg_sim_mean}
        return loss, stats


__all__ = ["NTXentLoss"]
