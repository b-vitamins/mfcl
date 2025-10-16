"""Evaluation utilities: linear probe and kNN on frozen features."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)

from mfcl.metrics.knn import knn_predict
from mfcl.utils.amp import AmpScaler


def _autocast(enabled: bool):
    """Return a CUDA autocast context or a no-op when disabled."""

    if not enabled:
        return nullcontext()
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore[attr-defined]

        return amp_autocast(device_type="cuda")
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast

        return cuda_autocast()


def _accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1, 5)
) -> Tuple[float, float]:
    """Compute top-k accuracies, clamping k to num classes.

    Always returns a 2-tuple corresponding to the requested (1, 5),
    but if the number of classes is < 5, the top-5 value is computed
    as top-C where C is the number of classes.
    """
    B = target.size(0)
    C = output.size(1)
    req = list(topk)
    ks = [min(int(k), C) for k in req]
    maxk = max(ks)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float((correct_k / B).item()))
    # Ensure we always return two values matching (1,5)
    if len(res) == 1:
        res.append(res[0])
    return res[0], res[1]


class LinearProbe:
    """Train a linear classifier on frozen encoder features."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        device: torch.device,
        lr: float = 30.0,
        epochs: int = 90,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        milestones: Tuple[int, int] = (30, 60),
        batch_size_override: Optional[int] = None,
        feature_norm: bool = False,
        amp: bool = False,
        use_scaler: bool = False,
    ) -> None:
        self.encoder = encoder
        self.encoder.eval()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.device = device
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.milestones = milestones
        if batch_size_override is not None and int(batch_size_override) <= 0:
            raise ValueError("batch_size_override must be a positive integer")
        self.batch_size_override = batch_size_override
        self.feature_norm = bool(feature_norm)
        self.amp = bool(amp)
        self.head = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        self.use_scaler = bool(use_scaler)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train linear head and report top1/top5 on val.

        Returns:
            {'top1': float, 'top5': float, 'loss': float}
        """
        train_loader = self._prepare_loader(train_loader, shuffle_default=True)
        val_loader = self._prepare_loader(val_loader, shuffle_default=False)

        opt = torch.optim.SGD(
            self.head.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=list(self.milestones), gamma=0.1
        )
        loss_meter = 0.0
        n_batches = 0
        amp_enabled = self.amp and torch.cuda.is_available()
        scaler = AmpScaler(enabled=amp_enabled and self.use_scaler)

        for epoch in range(1, self.epochs + 1):
            self.head.train()
            for images, targets in train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.no_grad():
                    with _autocast(amp_enabled):
                        feats = self.encoder(images)
                    feats = feats.detach()
                    if self.feature_norm:
                        feats = torch.nn.functional.normalize(feats, dim=1)
                opt.zero_grad(set_to_none=True)
                with _autocast(amp_enabled):
                    logits = self.head(feats)
                    loss = F.cross_entropy(logits, targets)
                if scaler.is_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                loss_meter += float(loss.item())
                n_batches += 1
            sched.step()

        # Validation
        self.head.eval()
        top1_sum = 0.0
        top5_sum = 0.0
        count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with _autocast(amp_enabled):
                    feats = self.encoder(images)
                if self.feature_norm:
                    feats = torch.nn.functional.normalize(feats, dim=1)
                with _autocast(amp_enabled):
                    logits = self.head(feats)
                t1, t5 = _accuracy(logits, targets, topk=(1, 5))
                top1_sum += t1 * images.size(0)
                top5_sum += t5 * images.size(0)
                count += images.size(0)
        top1 = top1_sum / max(1, count)
        top5 = top5_sum / max(1, count)
        avg_loss = loss_meter / max(1, n_batches)
        return {"top1": float(top1), "top5": float(top5), "loss": float(avg_loss)}

    def _prepare_loader(self, loader: DataLoader, *, shuffle_default: bool) -> DataLoader:
        """Best-effort DataLoader clone honoring ``batch_size_override``."""
        if self.batch_size_override is None:
            return loader
        if not isinstance(loader, DataLoader):
            raise TypeError(
                "batch_size_override requires torch.utils.data.DataLoader instances"
            )
        override = int(self.batch_size_override)
        if loader.batch_size == override:
            return loader
        if isinstance(loader.dataset, IterableDataset):
            raise ValueError(
                "batch_size_override is not supported for IterableDataset loaders"
            )

        sampler = getattr(loader, "sampler", None)
        sampler_arg = None
        shuffle_flag = bool(shuffle_default)
        if sampler is not None:
            if isinstance(sampler, (RandomSampler, SequentialSampler)):
                shuffle_flag = isinstance(sampler, RandomSampler)
            else:
                sampler_arg = sampler
                shuffle_flag = False

        kwargs: Dict[str, Any] = {
            "dataset": loader.dataset,
            "batch_size": override,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "collate_fn": loader.collate_fn,
            "persistent_workers": getattr(loader, "persistent_workers", False)
            if loader.num_workers > 0
            else False,
            "timeout": loader.timeout,
        }
        if loader.worker_init_fn is not None:
            kwargs["worker_init_fn"] = loader.worker_init_fn
        if loader.generator is not None:
            kwargs["generator"] = loader.generator
        prefetch = getattr(loader, "prefetch_factor", None)
        if prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
        pin_memory_device = getattr(loader, "pin_memory_device", None)
        if pin_memory_device:
            kwargs["pin_memory_device"] = pin_memory_device

        if sampler_arg is not None:
            kwargs["sampler"] = sampler_arg
        else:
            kwargs["shuffle"] = shuffle_flag

        return DataLoader(**kwargs)


class KNNEval:
    """kNN eval on frozen features with a feature bank."""

    def __init__(
        self, k: int = 200, temperature: float = 0.07, normalize: bool = True
    ) -> None:
        self.k = int(k)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    @torch.no_grad()
    def build_bank(
        self, encoder: nn.Module, loader: DataLoader, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features [N,D], labels [N])."""
        encoder.eval()
        feats_list = []
        labels_list = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feats = encoder(images)
            if self.normalize:
                feats = torch.nn.functional.normalize(feats, dim=1)
            feats_list.append(feats.cpu())
            labels_list.append(targets.cpu())
        feats_all = torch.cat(feats_list, dim=0).to(device)
        labels_all = torch.cat(labels_list, dim=0).to(device)
        return feats_all, labels_all

    @torch.no_grad()
    def evaluate(
        self,
        encoder: nn.Module,
        loader: DataLoader,
        bank_feats: torch.Tensor,
        bank_labels: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """Return {'top1': float, 'top5': float}."""
        encoder.eval()
        top1_sum = 0.0
        top5_sum = 0.0
        count = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feats = encoder(images)
            if self.normalize:
                feats = torch.nn.functional.normalize(feats, dim=1)
            probs = knn_predict(
                feats, bank_feats, bank_labels, k=self.k, temperature=self.temperature
            )
            # _accuracy operates on logits or probabilities; kNN already outputs probabilities.
            t1, t5 = _accuracy(probs, targets, topk=(1, 5))
            top1_sum += t1 * images.size(0)
            top5_sum += t5 * images.size(0)
            count += images.size(0)
        return {
            "top1": float(top1_sum / max(1, count)),
            "top5": float(top5_sum / max(1, count)),
        }


__all__ = ["LinearProbe", "KNNEval"]
