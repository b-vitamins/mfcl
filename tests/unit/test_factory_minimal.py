import pytest
import torch

import mfcl.core.factory as F
from mfcl.core.config import (
    Config,
    DataConfig,
    AugConfig,
    ModelConfig,
    MethodConfig,
    OptimConfig,
    TrainConfig,
)
from tests.helpers.nets import TinyEncoder


class DummyBadLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


def test_build_method_with_monkeypatched_encoder(monkeypatch):
    # monkeypatch build_encoder to avoid heavy backbones
    monkeypatch.setattr(
        F, "build_encoder", lambda cfg: TinyEncoder(cfg.model.encoder_dim)
    )
    cfg = Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(img_size=32),
        model=ModelConfig(
            encoder="resnet18", encoder_dim=16, projector_hidden=32, projector_out=8
        ),
        method=MethodConfig(name="simclr", temperature=0.1),
        optim=OptimConfig(),
        train=TrainConfig(),
    )
    m = F.build_method(cfg)
    assert isinstance(m, torch.nn.Module)


def test_build_loss_enforces_contract(monkeypatch):
    # Replace the loss constructor for NT-Xent with a bad one
    from mfcl.core.factory import LOSS_REGISTRY

    key = "ntxent"
    original = LOSS_REGISTRY.get(key)
    try:
        LOSS_REGISTRY._map[key] = DummyBadLoss  # type: ignore[attr-defined]
        cfg = Config(
            data=DataConfig(root="/tmp"),
            aug=AugConfig(img_size=32),
            model=ModelConfig(
                encoder="resnet18", encoder_dim=16, projector_hidden=32, projector_out=8
            ),
            method=MethodConfig(name="simclr", temperature=0.1),
            optim=OptimConfig(),
            train=TrainConfig(),
        )
        with pytest.raises(TypeError):
            F.build_loss(cfg)
    finally:
        LOSS_REGISTRY._map[key] = original  # type: ignore[attr-defined]
