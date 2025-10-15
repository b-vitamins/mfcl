"""Configurable MLP projector head.

Used by SimCLR, MoCo v2, BYOL, SimSiam, SwAV, Barlow Twins, and VICReg.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(kind: Literal["relu", "gelu"]) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {kind}")


class Projector(nn.Module):
    """Multi-layer perceptron used to map encoder features to embedding space."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        use_bn: bool = True,
        bn_last: bool = False,
        activation: Literal["relu", "gelu"] = "relu",
        norm_out: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize the projector.

        Args:
            in_dim: Input feature dimension.
            hidden_dim: Hidden width for intermediate layers.
            out_dim: Output embedding dimension.
            num_layers: Total linear layers (>= 2 recommended).
            use_bn: Apply BatchNorm1d after each hidden linear.
            bn_last: If True, apply BN on the final layer output (before norm_out).
            activation: Nonlinearity for hidden layers.
            norm_out: If True, L2-normalize output along dim=1.
            dropout: Dropout p in hidden layers (0 disables).
            bias: Linear layer bias flags.

        Raises:
            ValueError: If num_layers < 1 or dims invalid.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim, hidden_dim and out_dim must be > 0")

        self.out_dim = int(out_dim)
        self.norm_out = bool(norm_out)

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if bn_last:
                layers.append(nn.BatchNorm1d(out_dim))
        else:
            # Hidden layers
            for i in range(num_layers - 1):
                in_f = in_dim if i == 0 else hidden_dim
                layers.append(nn.Linear(in_f, hidden_dim, bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(_make_activation(activation))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(p=float(dropout)))
            # Final layer
            layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
            if bn_last:
                layers.append(nn.BatchNorm1d(out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation: Literal["relu", "gelu"]) -> None:
        # Hidden linears: Kaiming normal; final linear: Xavier normal
        last_linear: nn.Linear | None = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if m is last_linear:
                    nn.init.xavier_normal_(m.weight)
                else:
                    # PyTorch kaiming_normal_ supports 'relu' and 'leaky_relu'. Use relu as proxy for gelu.
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize BN to sane defaults
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        """Return out_dim."""
        return self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features into embedding space.

        Args:
            x: [B, in_dim] features.

        Returns:
            [B, out_dim] projections (optionally L2-normalized).
        """
        out = self.net(x)
        if self.norm_out:
            out = F.normalize(out, dim=1)
        return out


__all__ = ["Projector"]
