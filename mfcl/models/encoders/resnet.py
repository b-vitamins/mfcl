"""Torchvision ResNet encoders with pooled feature outputs.

Thin wrappers around torchvision ResNets that return a single global feature
vector per image (or the last feature map, if requested).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


_VARIANT_TO_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


def _create_tv_resnet(variant: str, pretrained: bool) -> nn.Module:
    """Create a torchvision resnet with robust handling across versions.

    Args:
        variant: One of 'resnet18', 'resnet34', 'resnet50'.
        pretrained: Whether to load pretrained weights.

    Returns:
        A torchvision ResNet instance.

    Raises:
        ValueError: If variant is unsupported.
    """
    from torchvision import models  # type: ignore

    if variant not in _VARIANT_TO_DIM:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

    ctor = getattr(models, variant)
    # Newer torchvision uses weights enums; older uses pretrained=True/False
    if pretrained:
        try:
            weights_enum = getattr(models, f"{variant}_Weights")
            try:
                # Prefer default weights when available.
                weights = getattr(weights_enum, "DEFAULT")
            except Exception:
                # Fallback to a likely imagenet weights variant.
                weights = list(weights_enum)[0]
            return ctor(weights=weights)
        except Exception:
            return ctor(pretrained=True)  # type: ignore[call-arg]
    else:
        try:
            return ctor(weights=None)
        except Exception:
            return ctor(pretrained=False)  # type: ignore[call-arg]


class ResNetEncoder(nn.Module):
    """ResNet encoder wrapper that returns pooled features.

    Removes the classification head and applies global average pooling.
    """

    backbone: nn.Module

    def __init__(
        self,
        variant: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        pretrained: bool = False,
        train_bn: bool = True,
        norm_feat: bool = True,
        pool: Literal["avg", "identity"] = "avg",
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
        in_channels: int = 3,
    ) -> None:
        """Construct a ResNet encoder.

        Args:
            variant: ResNet variant to instantiate.
            pretrained: Whether to load torchvision pretrained weights.
            train_bn: If ``False``, BatchNorm/SyncBatchNorm/GroupNorm layers are
                set to eval mode and their parameters are frozen.
            norm_feat: If ``True``, L2-normalize pooled output features.
            pool: ``"avg"`` applies global average pooling; ``"identity"``
                returns the spatial feature map.
            drop_path_rate: Drop path regularization rate. Torchvision ResNets do
                not support stochastic depth; if non-zero a warning is emitted
                and the value is ignored.
            freeze_backbone: If ``True``, all backbone parameters are frozen.
            in_channels: Number of input channels expected by ``forward``. When
                loading pretrained weights and the value differs from ``3``, the
                first convolution kernel is adapted using channel-mean
                initialization (for ``<=3`` channels) or repetition (for more
                channels).

        Raises:
            ValueError: If ``variant`` is unsupported or ``pool`` is invalid.

        Returns:
            None
        """
        super().__init__()
        if pool not in {"avg", "identity"}:
            raise ValueError("pool must be 'avg' or 'identity'")
        if drop_path_rate:
            warnings.warn(
                "drop_path_rate is ignored for torchvision ResNets.",
                UserWarning,
                stacklevel=2,
            )
        self.variant = variant
        self.norm_feat = bool(norm_feat)
        self.pool_mode = pool
        self.in_channels = int(in_channels)
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        self._feat_dim = _VARIANT_TO_DIM.get(variant, 0)

        backbone: nn.Module = _create_tv_resnet(variant, pretrained=pretrained)
        # Remove classifier head to expose features
        backbone.fc = nn.Identity()
        if self.in_channels != getattr(backbone.conv1, "in_channels", self.in_channels):
            old_conv: nn.Conv2d = backbone.conv1  # type: ignore[assignment]
            new_conv = nn.Conv2d(
                self.in_channels,
                old_conv.out_channels,
                kernel_size=_to_pair(old_conv.kernel_size),
                stride=_to_pair(old_conv.stride),
                padding=_to_pair(old_conv.padding),
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    weight = old_conv.weight.data
                    if self.in_channels == 1:
                        adapted = weight.mean(dim=1, keepdim=True)
                    elif self.in_channels > weight.shape[1]:
                        repeat = -(-self.in_channels // weight.shape[1])
                        adapted = weight.repeat(1, repeat, 1, 1)[
                            :, : self.in_channels, :, :
                        ]
                    else:
                        base = weight.mean(dim=1, keepdim=True)
                        adapted = base.repeat(1, self.in_channels, 1, 1)
                    new_conv.weight.data.copy_(adapted)
            backbone.conv1 = new_conv
        self.backbone: nn.Module = backbone
        self.pool = nn.AdaptiveAvgPool2d(1) if pool == "avg" else nn.Identity()

        if not train_bn:
            for m in self.backbone.modules():
                if isinstance(
                    m,
                    (
                        nn.modules.batchnorm._BatchNorm,
                        nn.SyncBatchNorm,
                        nn.GroupNorm,
                    ),
                ):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """Feature dimension after pooling.

        Returns:
            int: Output feature dimensionality.
        """
        return int(self._feat_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward input through the ResNet feature extractor.

        The implementation assumes the wrapped module exposes torchvision-style
        attributes (``conv1``, ``bn1``, ``layer1`` ... ``layer4``).

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: The final convolutional feature map prior to pooling.
        """
        m: object = self.backbone
        x = getattr(m, "conv1")(x)  # type: ignore[no-any-call]
        x = getattr(m, "bn1")(x)  # type: ignore[no-any-call]
        x = getattr(m, "relu")(x)  # type: ignore[no-any-call]
        x = getattr(m, "maxpool")(x)  # type: ignore[no-any-call]
        x = getattr(m, "layer1")(x)  # type: ignore[no-any-call]
        x = getattr(m, "layer2")(x)  # type: ignore[no-any-call]
        x = getattr(m, "layer3")(x)  # type: ignore[no-any-call]
        x = getattr(m, "layer4")(x)  # type: ignore[no-any-call]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute encoder features.

        Args:
            x: Input tensor with shape ``[B, C, H, W]`` where ``C`` matches the
                ``in_channels`` argument provided at construction.

        Returns:
            torch.Tensor: If ``pool == 'avg'`` returns ``[B, D]`` where
                ``D == feature_dim``. If ``pool == 'identity'`` returns the
                penultimate feature map ``[B, D, H', W']``.

        Notes:
            If ``norm_feat`` is ``True`` and ``pool == 'avg'``, the output is
            L2-normalized across the feature dimension.
        """
        feats = self._forward_features(x)
        if isinstance(self.pool, nn.Identity):
            return feats
        pooled = self.pool(feats).flatten(1)
        if self.norm_feat:
            pooled = F.normalize(pooled, dim=1)
        return pooled

    def set_train_bn(self, train: bool) -> None:
        """Toggle training mode and gradients for normalization layers.

        Args:
            train: ``True`` enables BatchNorm statistics updates and gradients;
                ``False`` freezes them. GroupNorm layers only have their
                gradients toggled as they are stateless with respect to
                train/eval mode.

        Returns:
            None
        """
        for m in self.backbone.modules():
            if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                m.train(train)
                for p in m.parameters(recurse=False):
                    p.requires_grad = train
            elif isinstance(m, nn.GroupNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad = train


def make_resnet18(**kwargs) -> ResNetEncoder:
    """Factory helper for ResNet-18 encoder."""
    return ResNetEncoder(variant="resnet18", **kwargs)


def make_resnet34(**kwargs) -> ResNetEncoder:
    """Factory helper for ResNet-34 encoder."""
    return ResNetEncoder(variant="resnet34", **kwargs)


def make_resnet50(**kwargs) -> ResNetEncoder:
    """Factory helper for ResNet-50 encoder."""
    return ResNetEncoder(variant="resnet50", **kwargs)


__all__ = [
    "ResNetEncoder",
    "make_resnet18",
    "make_resnet34",
    "make_resnet50",
]
def _to_pair(value: Any) -> tuple[int, int]:
    """Return a 2-tuple representation for Conv2d arguments."""

    first, second = _pair(value)
    return int(first), int(second)
