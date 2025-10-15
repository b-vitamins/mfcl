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

        Returns:
            None
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
                self._weight.copy_(F.normalize(self._weight, dim=1, eps=1e-12))

        self._bias = nn.Parameter(torch.zeros(num_prototypes)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        """Return prototype weights of shape [num_prototypes, feat_dim].

        Returns:
            torch.Tensor: Learnable prototype weight matrix.
        """
        return self._weight

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """L2-normalize each prototype vector in-place (if normalize=True).

        Returns:
            None
        """
        if self.normalize:
            self._weight.copy_(F.normalize(self._weight, dim=1, eps=1e-12))

    def set_temperature(self, temperature: float) -> None:
        """Update the softmax temperature.

        Args:
            temperature: Positive scaling factor applied to logits.

        Raises:
            ValueError: If ``temperature`` is not strictly positive.

        Returns:
            None
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.t = float(temperature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute prototype logits.

        Args:
            z: Input tensor of shape ``[B, D]``. For stable training, callers
                should provide L2-normalized features.

        Returns:
            torch.Tensor: Logits of shape ``[B, num_prototypes]`` equal to
                ``(z @ W^T) / temperature`` plus the optional bias term.
        """
        # Optional debugging aid for callers that forget to normalize inputs.
        # if __debug__:
        #     with torch.no_grad():
        #         norms = z.norm(dim=1).mean().item()
        #         if not (0.9 <= norms <= 1.1):
        #             raise ValueError("SwAVPrototypes expects L2-normalized inputs.")
        logits = (z @ self._weight.t()) / self.t
        if self._bias is not None:
            logits = logits + self._bias
        return logits

    def __repr__(self) -> str:
        """Return a readable summary of the prototype configuration.

        Returns:
            str: Human-readable description of the prototypes.
        """
        return (
            f"SwAVPrototypes(k={self._weight.size(0)}, d={self._weight.size(1)}, "
            f"t={self.t}, norm={self.normalize})"
        )


__all__ = ["SwAVPrototypes"]
