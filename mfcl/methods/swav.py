"""SwAV method with multi-crop, projector, prototypes, and swapped assignment loss."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import torch
import torch.nn as nn

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.models.prototypes.swavproto import SwAVPrototypes
from mfcl.losses.swavloss import SwAVLoss


class SwAV(BaseMethod):
    """SwAV with multi-crop and prototype assignments."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: Projector,
        prototypes: SwAVPrototypes,
        temperature: float = 0.1,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        normalize_input: bool = True,
        use_float32_for_sinkhorn: bool = True,
        sinkhorn_tol: float = 1e-3,
        sinkhorn_max_iters: int = 100,
        codes_queue_size: int = 0,
        loss_fp32: bool = True,
    ) -> None:
        """Construct SwAV.

        Args:
            encoder: Backbone.
            projector: Projects to prototype space dim D_p.
            prototypes: SwAVPrototypes with K clusters.
            temperature: Softmax temperature for logits before loss.
            epsilon: Sinkhorn entropic regularization.
            sinkhorn_iters: Number of Sinkhorn iterations.
            normalize_input: Whether to L2-normalize projector outputs before prototypes.
            use_float32_for_sinkhorn: Cast Sinkhorn computation to float32 for stability.
            sinkhorn_tol: Early-stop tolerance for Sinkhorn balancing.
            sinkhorn_max_iters: Maximum iterations for Sinkhorn refinement.
            codes_queue_size: Optional number of past logits (rows) to retain per
                code crop for assignment stabilization. ``0`` disables the queue.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.prototypes = prototypes
        self.loss_fn = SwAVLoss(
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            temperature=temperature,
            use_float32_for_sinkhorn=use_float32_for_sinkhorn,
            sinkhorn_tol=sinkhorn_tol,
            sinkhorn_max_iters=sinkhorn_max_iters,
            force_fp32=loss_fp32,
        )
        self.normalize_input = normalize_input
        self.codes_queue_size = max(int(codes_queue_size), 0)
        self._codes_queue: Dict[int, Deque[torch.Tensor]] = {}
        self._codes_queue_rows: Dict[int, int] = {}

    def forward_views(self, batch: Dict[str, Any]):
        crops: List[torch.Tensor] = batch["crops"]
        # Encode and project each crop independently
        zs = [self.projector(self.encoder(x)) for x in crops]
        if self.normalize_input:
            zs = [torch.nn.functional.normalize(z, dim=1) for z in zs]
        logits = [self.prototypes(z) for z in zs]  # list of [B, K]
        return logits, batch["code_crops"]

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]):
        logits, code_idx = proj
        queue_logits = self._gather_queue_logits(logits, code_idx)
        loss, stats = self.loss_fn(logits, code_idx, queue_logits=queue_logits)  # type: ignore[arg-type]
        stats["loss"] = loss
        self._enqueue_codes(logits, code_idx)
        return stats

    def on_optimizer_step(self) -> None:
        # Normalize prototype weights after each optimizer step
        self.prototypes.normalize_weights()

    def _gather_queue_logits(
        self, logits: List[torch.Tensor], code_idx: Any
    ) -> Dict[int, torch.Tensor]:
        if self.codes_queue_size <= 0:
            return {}
        gathered: Dict[int, torch.Tensor] = {}
        for idx in code_idx:
            stored = self._codes_queue.get(int(idx))
            if stored:
                tensor = torch.cat(list(stored), dim=0)
                gathered[int(idx)] = tensor.to(
                    device=logits[int(idx)].device, dtype=logits[int(idx)].dtype
                )
        return gathered

    def _enqueue_codes(self, logits: List[torch.Tensor], code_idx: Any) -> None:
        if self.codes_queue_size <= 0:
            return
        for idx in code_idx:
            key = int(idx)
            entry = logits[key].detach().to(torch.float32).cpu()
            queue = self._codes_queue.get(key)
            if queue is None:
                queue = deque(maxlen=self.codes_queue_size)
                self._codes_queue[key] = queue
            if entry.shape[0] == 0:
                self._codes_queue_rows[key] = len(queue)
                continue
            for row in entry.split(1, dim=0):
                queue.append(row)
            self._codes_queue_rows[key] = len(queue)


__all__ = ["SwAV"]
