"""timm backbone wrapper that returns pooled features.

Optional encoder that unifies timm models with the same feature contract as the
torchvision resnet wrapper. Import is guarded and will raise if timm is missing.
If ``global_pool == ""`` the encoder returns a feature map whose shape depends
on the backbone.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class TimmEncoder(nn.Module):
    """timm backbone wrapper that returns pooled features."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        train_bn: bool = True,
        norm_feat: bool = True,
        global_pool: str = "avg",
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        """Construct a timm encoder.

        Args:
            model_name: timm model name (e.g., ``"resnet18"`` or
                ``"convnext_tiny"``).
            pretrained: Whether to load pretrained weights if available.
            train_bn: If ``False``, BatchNorm/SyncBatchNorm/LayerNorm/GroupNorm
                layers are frozen.
            norm_feat: If ``True``, L2-normalize the pooled feature vectors.
            global_pool: Forwarded to ``timm.create_model(global_pool=...)``.
            drop_path_rate: Forwarded to ``timm.create_model``.
            freeze_backbone: If ``True``, disables gradients for all backbone
                parameters.

        Raises:
            ImportError: If timm is not installed.
            ValueError: If ``model_name`` is invalid or the feature dimension
                cannot be inferred.

        Returns:
            None
        """
        super().__init__()
        self.norm_feat = bool(norm_feat)
        self.global_pool = global_pool

        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("timm not installed") from e

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
            drop_path_rate=drop_path_rate,
        )

        # Infer feature dimension
        feat_dim = getattr(self.model, "num_features", None)
        if not isinstance(feat_dim, int) or feat_dim <= 0:
            # Probe with a dummy forward
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                param = next(self.model.parameters())
                dummy = torch.zeros(
                    1,
                    3,
                    160,
                    160,
                    device=param.device,
                    dtype=param.dtype,
                )
                out = self.model(dummy)
            self.model.train(was_training)
            if not isinstance(out, torch.Tensor) or out.ndim < 2:
                raise ValueError("Unable to infer feature dim from timm model output.")
            feat_dim = int(out.shape[1])
        self._feat_dim = int(feat_dim)

        if not train_bn:
            for m in self.model.modules():
                if isinstance(
                    m,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.SyncBatchNorm,
                        nn.LayerNorm,
                        nn.GroupNorm,
                    ),
                ):
                    m.eval()
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """Feature dimension after pooling.

        Returns:
            int: Output feature dimensionality.
        """
        return int(self._feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute encoder features.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` with ``C`` matching the
                model's expected number of channels.

        Returns:
            torch.Tensor: ``[B, D]`` if ``global_pool`` is non-empty. If
                ``global_pool == ""`` the return value follows timm's native
                output (e.g., a feature map).
        """
        out = self.model(x)
        # If pooled, timm returns [B, D]
        if out.ndim == 2 and self.norm_feat:
            out = F.normalize(out, dim=1)
        return out


__all__ = ["TimmEncoder"]
