"""Predictor head for BYOL/SimSiam.

Small MLP with a bottleneck: Linear -> [BN] -> Act -> Linear.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _make_activation(kind: Literal["relu", "gelu"]) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {kind}")


class Predictor(nn.Module):
    """BYOL/SimSiam predictor MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_bn: bool = True,
        activation: Literal["relu", "gelu"] = "relu",
        bias: bool = True,
    ) -> None:
        """Construct the predictor head.

        Args:
            in_dim: Input dimension (matches projector output).
            hidden_dim: Bottleneck width (e.g., 512 or 1024).
            out_dim: Output dimension (same as in_dim for symmetric heads).
            use_bn: Apply BN on hidden layer.
            activation: Nonlinearity for hidden layer.
            bias: Bias flags for Linear layers.
        """
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(_make_activation(activation))
        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        # Hidden linear: Kaiming normal; final linear: Xavier normal; biases zero
        last_linear: nn.Linear | None = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if m is last_linear:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, out_dim] predictions."""
        return self.net(x)


__all__ = ["Predictor"]
