"""MoCo v2 method with momentum key encoder and negative queue."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfcl.methods.base import BaseMethod
from mfcl.models.heads.projector import Projector
from mfcl.utils.ema import MomentumUpdater
from mfcl.utils.queue import RingQueue
from mfcl.utils import dist as dist_utils
from mfcl.telemetry.hardness import get_active_monitor
from mfcl.data.schema import extract_labels


class MoCo(BaseMethod):
    """MoCo v2 with momentum encoder and negative queue."""

    def __init__(
        self,
        encoder_q: nn.Module,
        encoder_k: nn.Module,
        projector_q: Projector,
        projector_k: Projector,
        temperature: float = 0.2,
        momentum: float = 0.999,
        queue_size: int = 65536,
        normalize: bool = True,
        cross_rank_queue: bool = False,
        queue_device: str | torch.device = "cpu",
        queue_dtype: str | torch.dtype = torch.float32,
        loss_fp32: bool = True,
    ) -> None:
        """Construct MoCo.

        Args:
            encoder_q: Online encoder (grad flows).
            encoder_k: Key encoder (EMA updated).
            projector_q: Projector for queries.
            projector_k: Projector for keys.
            temperature: InfoNCE temperature.
            momentum: EMA momentum for key encoder params.
            queue_size: Number of stored negatives.
            normalize: Normalize in the loss.
            cross_rank_queue: If True and using DDP, gather keys across ranks
                before updating the queue so all processes share negatives.
        """
        super().__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.projector_q = projector_q
        self.projector_k = projector_k
        self.updater = MomentumUpdater(self.encoder_q, self.encoder_k, momentum)

        if isinstance(queue_device, str):
            queue_device = torch.device(queue_device)
        if isinstance(queue_dtype, str):
            mapping = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "fp16": torch.float16,
                "float16": torch.float16,
            }
            key = queue_dtype.lower()
            if key not in mapping:
                raise ValueError(f"Unsupported queue dtype: {queue_dtype}")
            queue_dtype_t = mapping[key]
        else:
            queue_dtype_t = queue_dtype

        self.queue_device = queue_device
        self.queue_dtype = queue_dtype_t
        self.queue = RingQueue(
            dim=projector_q.output_dim,
            size=queue_size,
            device=self.queue_device,
            dtype=self.queue_dtype,
        )
        self.temperature = float(temperature)
        self.do_normalize = bool(normalize)
        self.cross_rank_queue = bool(cross_rank_queue)
        self.loss_force_fp32 = bool(loss_fp32)

    def on_train_start(self) -> None:
        self.updater.copy_params()

    def forward_views(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.projector_q(self.encoder_q(batch["view1"]))
        with torch.no_grad():
            k = self.projector_k(self.encoder_k(batch["view2"]))
        return q, k

    def compute_loss(self, *proj: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        q, k = proj
        extract_labels(batch)
        queue_keys_raw = self._queue_keys(k)
        queue_keys = self._format_queue_keys(queue_keys_raw)
        seeded_queue = False
        if len(self.queue) == 0:
            self.queue.enqueue(queue_keys)
            seeded_queue = True

        negatives = self.queue.get()

        loss, stats = self._info_nce_with(q, k, negatives)

        if not seeded_queue:
            self.queue.enqueue(queue_keys)

        stats["loss"] = loss
        stats["queue_len"] = torch.tensor(
            len(self.queue), device=q.device, dtype=torch.float32
        )
        stats["queue_capacity"] = torch.tensor(
            self.queue.size, device=q.device, dtype=torch.float32
        )
        return stats

    def on_optimizer_step(self) -> None:
        self.updater.update()

    def _info_nce_with(
        self, q: torch.Tensor, k: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute InfoNCE with explicitly provided negatives.

        The helper defensively moves all operands to the query device so future
        refactors cannot accidentally introduce cross-device matmuls.
        """
        if q.shape != k.shape or q.ndim != 2:
            raise ValueError("q and k must be 2D tensors with identical shapes")

        compute_dtype = torch.float32 if self.loss_force_fp32 else q.dtype
        qf = q.to(compute_dtype)
        kf = k.detach().to(compute_dtype)
        if self.do_normalize:
            qf = F.normalize(qf, dim=1)
            kf = F.normalize(kf, dim=1)

        if negatives.numel() > 0:
            negf = negatives.detach().to(
                device=q.device, dtype=compute_dtype, non_blocking=True
            )
        else:
            negf = qf.new_empty((0, qf.size(1)))

        pos = torch.sum(qf * kf, dim=1, keepdim=True)
        if negf.numel() > 0:
            neg_logits = qf @ negf.t()
            neg_sim_mean = neg_logits.mean().detach()
            monitor = get_active_monitor()
            if monitor is not None:
                monitor.add_negatives(neg_logits.detach().to(torch.float32))
        else:
            neg_logits = qf.new_empty((qf.size(0), 0))
            neg_sim_mean = qf.new_zeros(())

        logits = torch.cat([pos, neg_logits], dim=1) / self.temperature
        labels = torch.zeros(qf.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        if loss.dtype != torch.float32:
            loss = loss.to(torch.float32)

        stats: Dict[str, torch.Tensor] = {
            "pos_sim": pos.mean().detach(),
            "neg_sim_mean": neg_sim_mean,
        }
        return loss, stats

    def _queue_keys(self, k: torch.Tensor) -> torch.Tensor:
        gather_dtype = torch.float32 if self.loss_force_fp32 else k.dtype
        keys = k.detach().to(gather_dtype)
        if self.cross_rank_queue and dist_utils.get_world_size() > 1:
            keys = dist_utils.all_gather_tensor(keys)
        return keys.cpu()

    def _format_queue_keys(self, keys: torch.Tensor) -> torch.Tensor:
        formatted = keys.to(device=self.queue_device)
        if self.do_normalize:
            formatted = F.normalize(formatted, dim=1)
        if formatted.dtype != self.queue_dtype:
            formatted = formatted.to(dtype=self.queue_dtype)
        return formatted


__all__ = ["MoCo"]
