import warnings

import pytest
import torch
from omegaconf import OmegaConf

import train
from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)


@pytest.mark.skipif(not hasattr(torch.backends, "cudnn"), reason="cuDNN backend unavailable")
def test_configure_cudnn_benchmark_warns_when_deterministic(monkeypatch):
    original = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train._configure_cudnn_benchmark(True, True)
        assert not torch.backends.cudnn.benchmark
        assert any("cudnn_bench" in str(w.message) for w in caught)
    finally:
        torch.backends.cudnn.benchmark = original


@pytest.mark.skipif(not hasattr(torch.backends, "cudnn"), reason="cuDNN backend unavailable")
def test_configure_cudnn_benchmark_respects_nondeterministic(monkeypatch):
    original = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train._configure_cudnn_benchmark(False, True)
        assert torch.backends.cudnn.benchmark
        assert not caught
    finally:
        torch.backends.cudnn.benchmark = original


class _StopExecution(Exception):
    """Sentinel exception to abort _hydra_entry early during tests."""


@pytest.mark.skipif(not hasattr(torch.backends, "cudnn"), reason="cuDNN backend unavailable")
def test_hydra_entry_uses_deterministic_benchmark_guard(monkeypatch):
    cfg_struct = Config(
        data=DataConfig(root="/tmp/data", name="imagenet"),
        aug=AugConfig(local_crops=0),
        model=ModelConfig(),
        method=MethodConfig(),
        optim=OptimConfig(),
        train=TrainConfig(cudnn_bench=True, save_dir=""),
    )
    cfg = OmegaConf.structured(cfg_struct)

    monkeypatch.setattr(train, "init_distributed", lambda: None)
    monkeypatch.setattr(train, "get_world_size", lambda: 1)
    monkeypatch.setattr(train, "get_local_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    called = {}

    def fake_configure(deterministic: bool, bench_requested: bool) -> None:
        called["args"] = (deterministic, bench_requested)
        raise _StopExecution

    monkeypatch.setattr(train, "_configure_cudnn_benchmark", fake_configure)

    with pytest.raises(_StopExecution):
        train._hydra_entry.__wrapped__(cfg)

    assert called["args"] == (True, True)
