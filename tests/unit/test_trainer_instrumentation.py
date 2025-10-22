from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mfcl.engines.trainer import Trainer
from mfcl.engines.trainer_options import TrainerOptions
from mfcl.telemetry.timers import StepTimer
from mfcl.utils.consolemonitor import ConsoleMonitor


class _ConstantLossMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return self.step(batch)

    def step(self, batch):
        x = batch["x"].to(self.weight.device)
        loss = (self.weight * x).mean()
        return {"loss": loss}


class _RecordingMemoryMonitor:
    def __init__(self) -> None:
        self.update_calls: list[tuple[int, int, int]] = []
        self.record_calls: list[tuple[int, int]] = []

    def update_step_context(self, *, epoch: int, step_index: int, global_step: int) -> None:
        self.update_calls.append((epoch, step_index, global_step))

    def record_step_snapshot(self, *, epoch: int, global_step: int) -> None:
        self.record_calls.append((epoch, global_step))


class _RecordingHardnessMonitor:
    def __init__(self) -> None:
        self.begin_calls: list[tuple[int, int]] = []
        self.end_calls = 0

    def begin_step(self, *, epoch: int, step: int) -> None:
        self.begin_calls.append((epoch, step))

    def end_step(self) -> None:
        self.end_calls += 1


def _make_loader(num_batches: int = 1):
    class _Loader:
        def __iter__(self):
            for _ in range(num_batches):
                yield {"x": torch.ones(4, 1)}

        def __len__(self):
            return num_batches

    return _Loader()


def test_trainer_epoch_with_instrumentation(tmp_path: Path) -> None:
    method = _ConstantLossMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.0)
    timer = StepTimer(
        enabled=True,
        log_path=tmp_path / "timer.csv",
        is_main=True,
        warmup_steps=0,
        sample_rate=1,
    )
    memory_monitor = _RecordingMemoryMonitor()
    hardness_monitor = _RecordingHardnessMonitor()

    trainer = Trainer(
        method,
        optimizer,
        options=TrainerOptions(
            console=ConsoleMonitor(),
            timer=timer,
            memory_monitor=memory_monitor,
            hardness_monitor=hardness_monitor,
        ),
    )

    loader = _make_loader(num_batches=3)

    try:
        results = trainer.train_one_epoch(epoch=0, loader=loader)
    finally:
        timer.close()

    assert results["loss"] == pytest.approx(1.0)
    assert timer._step_counter == 3

    assert memory_monitor.update_calls == [
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
    ]
    assert memory_monitor.record_calls == [(0, 1), (0, 2), (0, 3)]

    assert hardness_monitor.begin_calls == [(0, 1), (0, 2), (0, 3)]
    assert hardness_monitor.end_calls == 3

