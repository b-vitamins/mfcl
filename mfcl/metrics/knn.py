"""kNN softmax classifier used for SSL evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_predict(
    features: torch.Tensor,  # [B, D]
    bank: torch.Tensor,  # [N, D]
    bank_labels: torch.Tensor,  # [N]
    k: int = 200,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Return class probabilities [B, C] using kNN with temperature-smoothed votes.

    Steps:
      - Compute cosine similarity between features and bank: sim = features @ bank.T
      - For each row, take top-k indices/values.
      - Convert neighbors' labels to one-hot and weight by softmax(sim / temperature).
      - Sum weights per class, then L1-normalize across classes.
    """
    device = features.device
    feats = features.to(torch.float32)
    bank = bank.to(device=device, dtype=torch.float32)
    bank_labels = bank_labels.to(device)
    B, D = feats.shape
    N = bank.shape[0]
    if N == 0:
        raise ValueError("Empty feature bank")
    k = int(min(max(1, k), N))

    # Cosine similarity (normalize here to ensure robust behavior)
    feats = F.normalize(feats, dim=1)
    bank = F.normalize(bank, dim=1)
    sim = feats @ bank.t()  # [B,N]
    vals, idx = torch.topk(sim, k=k, dim=1)  # [B,k]
    weights = torch.softmax(vals / float(temperature), dim=1)  # [B,k]
    neighbor_labels = bank_labels[idx]  # [B,k]

    C = int(bank_labels.max().item()) + 1
    if (bank_labels < 0).any():
        raise ValueError("bank_labels must be non-negative integer class ids")
    one_hot = F.one_hot(neighbor_labels, num_classes=C).to(weights.dtype)  # [B,k,C]
    class_votes = (one_hot * weights.unsqueeze(-1)).sum(dim=1)  # [B,C]
    class_votes = class_votes / (class_votes.sum(dim=1, keepdim=True) + 1e-12)
    return class_votes


__all__ = ["knn_predict"]
