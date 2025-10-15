"""SwAV prototypes: learnable cluster centers with optional normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVPrototypes(nn.Module):
    """Prototype matrix for SwAV with optional L2-normalization."""

    def __init__(
        self,
        num_prototypes: int,
        feat_dim: int,
        normalize: bool = True,
        temperature: float = 1.0,
        bias: bool = False,
    ) -> None:
        """Construct prototype matrix.

        Args:
            num_prototypes: Number of clusters (e.g., 3000).
            feat_dim: Feature dimension of projector output.
            normalize: If True, L2-normalize prototype weights each step.
            temperature: Logit scaling factor (>0).
            bias: Whether to include a learnable bias per prototype.

        Raises:
            ValueError: If num_prototypes <= 0 or feat_dim <= 0 or temperature <= 0.
        """
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be > 0")
        if feat_dim <= 0:
            raise ValueError("feat_dim must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.normalize = bool(normalize)
        self.t = float(temperature)

        self._weight = nn.Parameter(torch.empty(num_prototypes, feat_dim))
        nn.init.normal_(self._weight, std=0.02)
        if self.normalize:
            with torch.no_grad():
                self._weight.copy_(F.normalize(self._weight, dim=1))

        self._bias = nn.Parameter(torch.zeros(num_prototypes)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        """Return prototype weights of shape [num_prototypes, feat_dim]."""
        return self._weight

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """L2-normalize each prototype vector in-place (if normalize=True)."""
        if self.normalize:
            self._weight.copy_(F.normalize(self._weight, dim=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute prototype logits.

        Args:
            z: [B, D] L2-normalized features.

        Returns:
            logits: [B, num_prototypes] = (z @ W^T) / temperature + bias
        """
        logits = (z @ self._weight.t()) / self.t
        if self._bias is not None:
            logits = logits + self._bias
        return logits


__all__ = ["SwAVPrototypes"]
