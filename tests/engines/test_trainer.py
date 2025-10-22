from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from mfcl.engines.trainer import Trainer
from mfcl.engines.trainer_options import TrainerOptions
from mfcl.engines.context import current_trainer, trainer_context
from mfcl.engines.hooks import Hook
from mfcl.runtime.budget import BudgetTracker
from mfcl.telemetry.timers import StepTimer
from mfcl.utils.consolemonitor import ConsoleMonitor


class _LinearMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(4, 1)

    def forward(self, batch):
        return self.step(batch)

    def step(self, batch):
        x = batch["view1"].flatten(1)
        return {"loss": self.net(x).mean()}

    def on_optimizer_step(self) -> None:  # pragma: no cover - optional hook
        return


def _toy_loader(batch_size: int = 2, steps: int = 4):
    class _Loader:
        def __iter__(self):
            for _ in range(steps):
                yield {
                    "view1": torch.randn(batch_size, 2, 2),
                    "view2": torch.randn(batch_size, 2, 2),
                    "index": torch.arange(batch_size),
                }

        def __len__(self):
            return steps

    return _Loader()


def _make_trainer(
    method: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    **option_kwargs,
) -> Trainer:
    return Trainer(
        method,
        optimizer,
        options=TrainerOptions(console=ConsoleMonitor(), **option_kwargs),
    )


def test_trainer_context_sets_current_trainer():
    method = _LinearMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    trainer = _make_trainer(method, optimizer)

    assert current_trainer() is None
    with trainer_context(trainer) as active:
        assert active is trainer
        assert current_trainer() is trainer
    assert current_trainer() is None


class _RecordingTimer(StepTimer):
    def __init__(self) -> None:
        self.begin_calls: List[Dict[str, int]] = []
        self.end_calls: List[Dict[str, float]] = []

    # The StepTimer API is fairly wide; implement the pieces the trainer exercises.
    def begin_step(self, *, epoch: int, step_index: int, global_step: int) -> None:  # type: ignore[override]
        self.begin_calls.append(
            {"epoch": epoch, "step_index": step_index, "global_step": global_step}
        )

    def record_data(self, data_time: float) -> None:  # pragma: no cover - timer contract
        return

    def end_step(self, step_time_s: float, ips: float) -> None:  # type: ignore[override]
        self.end_calls.append({"step_time_s": step_time_s, "ips": ips})

    # Context manager helpers used by the trainer.
    def range_forward(self):  # pragma: no cover - trivial
        from contextlib import nullcontext

        return nullcontext()

    range_backward = range_optimizer = range_misc = range_forward
    range_topr = range_beta_ctrl = range_forward

    def add_comm_ms(self, value: float) -> None:  # pragma: no cover - optional
        return


class _HookRecorder(Hook):
    def __init__(self) -> None:
        self.batch_metrics: List[Dict[str, float]] = []

    def on_batch_end(self, global_step: int, metrics: Dict[str, float]) -> None:
        self.batch_metrics.append(metrics)


def test_checkpoint_roundtrip(tmp_path: Path):
    method = _LinearMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    trainer = _make_trainer(
        method,
        optimizer,
        save_dir=str(tmp_path),
        keep_k=2,
    )
    loader = _toy_loader(steps=3)
    trainer.fit(loader, epochs=1, save_every=1, eval_every=10)

    checkpoint = Path(tmp_path) / "ckpt_ep0001.pt"
    assert checkpoint.exists()
    initial_step = trainer._global_step

    # New trainer should resume from the saved checkpoint and continue training.
    new_method = _LinearMethod()
    new_optimizer = torch.optim.SGD(new_method.parameters(), lr=0.1)
    resumed = _make_trainer(
        new_method,
        new_optimizer,
        save_dir=str(tmp_path),
    )
    next_epoch = resumed._checkpoint_manager.resume_from(str(checkpoint))
    assert next_epoch == 2
    torch.testing.assert_close(new_method.net.weight.detach(), method.net.weight.detach())

    resumed.fit(loader, epochs=2, resume_path=str(checkpoint), save_every=10, eval_every=10)
    assert resumed._global_step > initial_step


def test_budget_enforcer_stops(tmp_path: Path):
    method = _LinearMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    tracker = BudgetTracker("iso_tokens", {"max_tokens": 4})
    trainer = _make_trainer(
        method,
        optimizer,
        save_dir=str(tmp_path),
        budget_tracker=tracker,
    )
    loader = _toy_loader(batch_size=2, steps=10)
    trainer.fit(loader, epochs=5, eval_every=10, save_every=10)

    snapshot = tracker.snapshot()
    assert snapshot["totals"]["tokens"] == 4
    assert tracker.should_stop()
    assert trainer._global_step == 1


def test_telemetry_hooks_operate(tmp_path: Path):
    method = _LinearMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.01)
    timer = _RecordingTimer()
    hook = _HookRecorder()
    trainer = _make_trainer(
        method,
        optimizer,
        save_dir=str(tmp_path),
        hooks=hook,
        timer=timer,
    )
    loader = _toy_loader(steps=2)
    trainer.fit(loader, epochs=1, save_every=10, eval_every=10)

    assert timer.begin_calls, "timer.begin_step should be invoked"
    assert timer.end_calls, "timer.end_step should be invoked"
    assert hook.batch_metrics, "hooks should receive batch metrics"
    for metrics in hook.batch_metrics:
        for value in metrics.values():
            assert isinstance(value, float)
