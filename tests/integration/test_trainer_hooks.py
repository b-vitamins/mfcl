from __future__ import annotations

import torch

from mfcl.engines.hooks import Hook
from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class DummyMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(3 * 4 * 4, 1)

    def step(self, batch):
        x = batch["view1"].flatten(1)
        loss = self.net(x).mean()
        return {"loss": loss}

    def on_optimizer_step(self) -> None:
        """Hook required by the Trainer protocol."""
        # Tests rely on this method existing; the implementation is a no-op.
        return None


class RecordingHook(Hook):
    def __init__(self) -> None:
        self.train_state = None
        self.epoch_states: list[tuple[int, int | None]] = []
        self.batch_steps: list[int] = []
        self.batch_metrics: list[dict[str, float]] = []

    def on_train_start(self, state):
        self.train_state = dict(state)

    def on_epoch_start(self, epoch, state):
        self.epoch_states.append((epoch, state.get("global_step")))

    def on_batch_end(self, step, metrics):
        self.batch_steps.append(step)
        self.batch_metrics.append(dict(metrics))


def _loader(num_batches: int = 3, batch_size: int = 2):
    class _It:
        def __iter__(self):
            for _ in range(num_batches):
                yield {
                    "view1": torch.randn(batch_size, 3, 4, 4),
                    "view2": torch.randn(batch_size, 3, 4, 4),
                    "index": torch.arange(batch_size),
                }

        def __len__(self):
            return num_batches

    return _It()


def test_trainer_hook_receives_steps_and_metrics(tmp_path):
    method = DummyMethod()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    hook = RecordingHook()
    trainer = Trainer(
        method,
        optimizer,
        console=ConsoleMonitor(),
        hooks=hook,
        save_dir=str(tmp_path),
    )
    trainer.fit(_loader(), epochs=1)

    assert hook.train_state is not None
    assert hook.train_state.get("global_step") == 0
    assert hook.epoch_states == [(1, 0)]
    assert hook.batch_steps == [1, 2, 3]

    for metrics in hook.batch_metrics:
        assert "loss" in metrics and isinstance(metrics["loss"], float)
        assert "lr" in metrics and isinstance(metrics["lr"], float)

    ckpt = next(tmp_path.glob("ckpt_ep0001.pt"))
    state = torch.load(ckpt)
    assert state.get("global_step") == len(hook.batch_steps)
