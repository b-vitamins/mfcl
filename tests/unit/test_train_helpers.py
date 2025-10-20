import pytest
import train as train_module

from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from train import _maybe_autofill_byol_schedule_steps, _warn_beta_ctrl_requires_mixture


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


def test_warn_beta_ctrl_requires_mixture(monkeypatch):
    captured: dict[str, object] = {}

    def fake_warn(message, category=None, stacklevel=1):
        captured["message"] = message
        captured["category"] = category
        captured["stacklevel"] = stacklevel

    monkeypatch.setattr(train_module.warnings, "warn", fake_warn)
    _warn_beta_ctrl_requires_mixture()

    assert captured["message"] == (
        "runtime.beta_ctrl.enabled is true but runtime.mixture.enabled is false; "
        "beta control requires mixture statistics."
    )
    assert captured["category"] is RuntimeWarning
    assert captured["stacklevel"] == 2
