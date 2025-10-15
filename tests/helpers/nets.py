from __future__ import annotations

import torch
from torch import nn


class TinyEncoder(nn.Module):
    """Small CNN that outputs [B, out_dim] by global average pooling."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Use high-precision mean for improved invariance to flip (order of reduction)
        # to satisfy strict equality tolerance in tests.
        # Equivalent to global average pooling.
        x = x.to(torch.float64).mean(dim=(-2, -1), keepdim=True).to(torch.float32)
        return x.flatten(1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
