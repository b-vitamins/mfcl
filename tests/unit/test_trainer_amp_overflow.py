from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch

from mfcl.engines.trainer import Trainer
from mfcl.engines.trainer_options import TrainerOptions
from mfcl.utils.consolemonitor import ConsoleMonitor


class _OverflowScaler:
    def __init__(self) -> None:
        self.scaler = self
        self._scale = 128.0

    def autocast(self):
        return nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:  # pragma: no cover - no-op
        return

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        # Simulate overflow by skipping optimizer.step()
        return

    def update(self) -> None:
        self._scale = self._scale / 2.0

    def state_dict(self):  # pragma: no cover - trainer expects serialization support
        return {}

    def load_state_dict(self, state):  # pragma: no cover - unused in test
        return

    def get_scale(self) -> float:
        return self._scale


class _ToyMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def step(self, batch):
        x = batch["x"].to(self.weight.device)
        loss = (self.weight * x).mean()
        return {"loss": loss}

    def forward(self, batch):
        return self.step(batch)


class _SpyScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self):  # pragma: no cover - required by Trainer
        return {"steps": self.steps}

    def load_state_dict(self, state):  # pragma: no cover - unused in test
        self.steps = state.get("steps", 0)


def _make_loader(num_batches: int = 1):
    class _Loader:
        def __iter__(self):
            for _ in range(num_batches):
                yield {"x": torch.ones(4, 1)}

        def __len__(self):
            return num_batches

    return _Loader()


def test_scheduler_skips_when_amp_overflow(tmp_path: Path):
    method = _ToyMethod()
    opt = torch.optim.SGD(method.parameters(), lr=0.1)
    sched = _SpyScheduler(opt)
    scaler = _OverflowScaler()
    trainer = Trainer(
        method,
        opt,
        scheduler=sched,
        options=TrainerOptions(
            scaler=scaler,
            console=ConsoleMonitor(),
            save_dir=str(tmp_path / "overflow"),
            scheduler_step_on="batch",
        ),
    )

    before = method.weight.detach().clone()
    trainer.fit(_make_loader(1), epochs=1)
    after = method.weight.detach().clone()

    assert torch.allclose(before, after)
    assert sched.steps == 0
