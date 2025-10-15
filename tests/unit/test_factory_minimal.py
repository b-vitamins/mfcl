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


@pytest.mark.parametrize(
    "method_name, expect_predictor, expected_loss",
    [
        ("SIMCLR", False, "NTXentLoss"),
        ("MoCo", False, "MoCoContrastLoss"),
        ("BYOL", True, "BYOLLoss"),
        ("SimSiam", True, "SimSiamLoss"),
    ],
)
def test_factory_is_case_insensitive(monkeypatch, method_name, expect_predictor, expected_loss):
    monkeypatch.setattr(
        F, "build_encoder", lambda cfg: TinyEncoder(cfg.model.encoder_dim)
    )
    cfg = Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(img_size=32),
        model=ModelConfig(
            encoder="resnet18",
            encoder_dim=16,
            projector_hidden=32,
            projector_out=8,
            predictor_hidden=16,
            predictor_out=8,
        ),
        method=MethodConfig(name=method_name, temperature=0.2),
        optim=OptimConfig(),
        train=TrainConfig(),
    )

    heads = F.build_heads(cfg)
    assert ("predictor" in heads) is expect_predictor

    loss = F.build_loss(cfg)
    assert loss.__class__.__name__ == expected_loss

    # The original casing should remain untouched to avoid side effects.
    assert cfg.method.name == method_name
