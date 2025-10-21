from __future__ import annotations

import pytest
import torch
from torch import nn

import mfcl.core.factory as F
from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from mfcl.core.registry import Registry


def _make_cfg(encoder_key: str, *, norm_feat: bool = True) -> Config:
    return Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(img_size=8),
        model=ModelConfig(
            encoder=encoder_key,
            encoder_dim=4,
            projector_hidden=8,
            projector_out=4,
            predictor_hidden=4,
            predictor_out=4,
            norm_feat=norm_feat,
        ),
        method=MethodConfig(),
        optim=OptimConfig(),
        train=TrainConfig(),
    )


class _BaseDummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), 4)


def test_build_encoder_passes_supported_norm_feat(monkeypatch: pytest.MonkeyPatch) -> None:
    class NormAwareEncoder(_BaseDummyEncoder):
        def __init__(self, *, norm_feat: bool) -> None:
            super().__init__()
            self.received_norm = norm_feat

    registry = Registry("encoder-test")
    registry.add("norm-aware", NormAwareEncoder)
    monkeypatch.setattr(F, "ENCODER_REGISTRY", registry)

    cfg = _make_cfg("norm-aware", norm_feat=False)

    encoder = F.build_encoder(cfg)

    assert isinstance(encoder, NormAwareEncoder)
    assert encoder.received_norm is False


def test_build_encoder_skips_unsupported_optional_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    class BareEncoder(_BaseDummyEncoder):
        def __init__(self) -> None:
            super().__init__()

    registry = Registry("encoder-test")
    registry.add("bare", BareEncoder)
    monkeypatch.setattr(F, "ENCODER_REGISTRY", registry)

    cfg = _make_cfg("bare", norm_feat=True)

    encoder = F.build_encoder(cfg)

    assert isinstance(encoder, BareEncoder)


def test_build_encoder_errors_when_required_kwarg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoModelNameEncoder(_BaseDummyEncoder):
        def __init__(self) -> None:
            super().__init__()

    registry = Registry("encoder-test")
    registry.add("timm", NoModelNameEncoder)
    monkeypatch.setattr(F, "ENCODER_REGISTRY", registry)

    cfg = _make_cfg("timm:tiny")

    with pytest.raises(TypeError, match="model_name"):
        F.build_encoder(cfg)


def test_build_encoder_requires_timm_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTimm(_BaseDummyEncoder):
        def __init__(self, *, model_name: str) -> None:
            super().__init__()

    registry = Registry("encoder-test")
    registry.add("timm", DummyTimm)
    monkeypatch.setattr(F, "ENCODER_REGISTRY", registry)

    cfg = _make_cfg("timm")

    with pytest.raises(ValueError, match="model name"):
        F.build_encoder(cfg)

