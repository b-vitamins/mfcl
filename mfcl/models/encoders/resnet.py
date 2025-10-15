"""Torchvision ResNet encoders with pooled feature outputs.

Thin wrappers around torchvision ResNets that return a single global feature
vector per image (or the last feature map, if requested).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(
        self,
        variant: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        pretrained: bool = False,
        train_bn: bool = True,
        norm_feat: bool = True,
        pool: Literal["avg", "identity"] = "avg",
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        """Construct a ResNet encoder.

        Args:
            variant: ResNet variant.
            pretrained: Load torchvision pretrained weights.
            train_bn: If False, batch norm layers are eval-mode and frozen.
            norm_feat: If True, L2-normalize output features along dim=1.
            pool: 'avg' uses AdaptiveAvgPool2d(1); 'identity' returns feature map.
            drop_path_rate: Not used by torchvision resnets; kept for parity with timm.
            freeze_backbone: If True, sets requires_grad=False for all backbone params.

        Raises:
            ValueError: If variant not supported or pool not recognized.
        """
        super().__init__()
        if pool not in {"avg", "identity"}:
            raise ValueError("pool must be 'avg' or 'identity'")
        self.variant = variant
        self.norm_feat = bool(norm_feat)
        self.pool_mode = pool
        self._feat_dim = _VARIANT_TO_DIM.get(variant, 0)

        backbone = _create_tv_resnet(variant, pretrained=pretrained)
        # Remove classifier head to expose features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1) if pool == "avg" else nn.Identity()

        if not train_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension after pooling."""
        return int(self._feat_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Re-implement torchvision ResNet feature extraction explicitly.
        # Use dynamic attribute access to avoid strict typing issues with stubs.
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
        """Compute features.

        Args:
            x: Float tensor [B, 3, H, W], normalized by caller.

        Returns:
            If pool=='avg': [B, D] where D = feature_dim.
            If pool=='identity': [B, D, H', W'] feature map from penultimate layer.

        Notes:
            - If norm_feat is True and pool=='avg', returns L2-normalized [B, D].
        """
        feats = self._forward_features(x)
        if isinstance(self.pool, nn.Identity):
            return feats
        pooled = self.pool(feats).flatten(1)
        if self.norm_feat:
            pooled = F.normalize(pooled, dim=1)
        return pooled


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
