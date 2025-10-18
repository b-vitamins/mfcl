import pytest

from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from train import _maybe_autofill_byol_schedule_steps


def _make_cfg(**method_kwargs) -> Config:
    method = MethodConfig(name="byol", **method_kwargs)
    return Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(),
        model=ModelConfig(),
        method=method,
        optim=OptimConfig(),
        train=TrainConfig(),
    )


def test_autofill_sets_steps_when_missing():
    cfg = _make_cfg(byol_momentum_schedule="cosine", byol_momentum_schedule_steps=None)
    _maybe_autofill_byol_schedule_steps(cfg, steps_per_epoch=5)
    assert cfg.method.byol_momentum_schedule_steps == 5 * cfg.train.epochs


def test_autofill_noop_for_non_cosine():
    cfg = _make_cfg(byol_momentum_schedule="const", byol_momentum_schedule_steps=None)
    _maybe_autofill_byol_schedule_steps(cfg, steps_per_epoch=5)
    assert cfg.method.byol_momentum_schedule_steps is None


def test_autofill_requires_steps_for_cosine():
    cfg = _make_cfg(byol_momentum_schedule="cosine", byol_momentum_schedule_steps=None)
    with pytest.raises(ValueError):
        _maybe_autofill_byol_schedule_steps(cfg, steps_per_epoch=None)
