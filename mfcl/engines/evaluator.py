"""Evaluation utilities: linear probe and kNN on frozen features."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mfcl.metrics.knn import knn_predict


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
        self.batch_size_override = batch_size_override
        self.head = nn.Linear(self.feature_dim, self.num_classes).to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train linear head and report top1/top5 on val.

        Returns:
            {'top1': float, 'top5': float, 'loss': float}
        """
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

        for epoch in range(1, self.epochs + 1):
            self.head.train()
            for images, targets in train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.no_grad():
                    feats = self.encoder(images)
                    feats = feats.detach()
                logits = self.head(feats)
                loss = F.cross_entropy(logits, targets)
                opt.zero_grad(set_to_none=True)
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
                feats = self.encoder(images)
                logits = self.head(feats)
                t1, t5 = _accuracy(logits, targets, topk=(1, 5))
                top1_sum += t1 * images.size(0)
                top5_sum += t5 * images.size(0)
                count += images.size(0)
        top1 = top1_sum / max(1, count)
        top5 = top5_sum / max(1, count)
        avg_loss = loss_meter / max(1, n_batches)
        return {"top1": float(top1), "top5": float(top5), "loss": float(avg_loss)}


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
            t1, t5 = _accuracy(probs, targets, topk=(1, 5))
            top1_sum += t1 * images.size(0)
            top5_sum += t5 * images.size(0)
            count += images.size(0)
        return {
            "top1": float(top1_sum / max(1, count)),
            "top5": float(top5_sum / max(1, count)),
        }


__all__ = ["LinearProbe", "KNNEval"]
