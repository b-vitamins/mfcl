from pathlib import Path

import torch

from mfcl.runtime.budget import BudgetTracker
from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class _TinyMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(8 * 8 * 3, 1)

    def forward(self, batch):
        return self.step(batch)

    def step(self, batch):
        x = batch["view1"].flatten(1)
        y = self.net(x).mean()
        return {"loss": y}

    def on_optimizer_step(self) -> None:  # pragma: no cover - interface hook
        return


def _toy_loader(batch_size: int = 2, steps: int = 5):
    class _Loader:
        def __iter__(self):
            for _ in range(steps):
                yield {
                    "view1": torch.randn(batch_size, 3, 8, 8),
                    "view2": torch.randn(batch_size, 3, 8, 8),
                    "index": torch.arange(batch_size),
                }

        def __len__(self):
            return steps

    return _Loader()


def test_budget_time_stop():
    tracker = BudgetTracker("iso_time", {"max_minutes": 0.05})
    assert not tracker.should_stop()
    tracker.update(step_samples=0, step_wall_ms=1000)
    tracker.update(step_samples=0, step_wall_ms=1000)
    assert not tracker.should_stop()
    assert not tracker.would_exceed(step_wall_ms=500)
    tracker.update(step_samples=0, step_wall_ms=1000)
    assert tracker.should_stop()
    assert tracker.would_exceed(step_wall_ms=1000)


def test_budget_tokens_with_accumulation(tmp_path: Path):
    method = _TinyMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    tracker = BudgetTracker("iso_tokens", {"max_tokens": 8})
    trainer = Trainer(
        method,
        optimizer,
        console=ConsoleMonitor(),
        save_dir=str(tmp_path),
        accum_steps=2,
        budget_tracker=tracker,
    )
    loader = _toy_loader(batch_size=2, steps=5)
    trainer.fit(loader, epochs=3, eval_every=10, save_every=10)

    snapshot = tracker.snapshot()
    assert snapshot["totals"]["tokens"] == 8
    assert snapshot["totals"]["steps"] == 2
    assert tracker.should_stop()
