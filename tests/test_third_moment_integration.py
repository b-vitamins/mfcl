"""Integration checks for third-moment sketch wiring in the trainer."""

from __future__ import annotations

import torch

from mfcl.engines.trainer import Trainer
from mfcl.moments.third import ThirdMomentSketch
from mfcl.moments.third import get_active_sketch
from mfcl.mixture.context import get_active_estimator


class _FakeEstimator:
    enabled = True

    def __init__(self, pi: torch.Tensor, mu: torch.Tensor) -> None:
        self._last_stats = {"pi": pi, "mu": mu}
        self.update_calls = 0

    def update(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused side effect
        self.update_calls += 1

    def log_step(self, *, step: int, epoch: int) -> None:  # pragma: no cover - noop
        return


class _FakeMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch):  # type: ignore[override]
        sketch = get_active_sketch()
        est = get_active_estimator()
        assert sketch is not None and sketch.has_mean
        assert est is not None
        x = batch["x"]
        sketch.update(x)
        loss = (self.linear(x) ** 2).mean()
        return {"loss": loss}


def test_trainer_sets_sketch_mean_and_updates(tmp_path, monkeypatch):
    _ = tmp_path  # ensure fixture is available for potential logging paths
    pi = torch.tensor([0.6, 0.4], dtype=torch.float32)
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    estimator = _FakeEstimator(pi=pi, mu=mu)
    sketch = ThirdMomentSketch(enabled=True, is_main=False, rank=2)

    update_calls: list[torch.Tensor] = []
    mean_calls: list[torch.Tensor] = []

    orig_update = sketch.update
    orig_set_mean = sketch.set_mean

    def _wrapped_update(*args, **kwargs):
        update_calls.append(args[0])
        return orig_update(*args, **kwargs)

    def _wrapped_set_mean(mean: torch.Tensor) -> None:
        mean_calls.append(mean.clone())
        orig_set_mean(mean)

    monkeypatch.setattr(sketch, "update", _wrapped_update)
    monkeypatch.setattr(sketch, "set_mean", _wrapped_set_mean)

    method = _FakeMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)

    batch = {
        "x": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "index": torch.tensor([0, 1], dtype=torch.int64),
    }

    trainer = Trainer(
        method=method,
        optimizer=optimizer,
        mixture_estimator=estimator,
        third_moment_sketch=sketch,
        save_dir=None,
        log_interval=1,
    )

    trainer.fit([batch], epochs=1, save_every=2)

    assert sketch.has_mean
    assert mean_calls, "set_mean was not invoked"
    expected_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
    assert torch.allclose(mean_calls[-1], expected_mu)
    assert update_calls, "update was not invoked"
