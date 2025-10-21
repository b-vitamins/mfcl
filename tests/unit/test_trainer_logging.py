import logging
from pathlib import Path

import torch

from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class _MethodWithHooks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    # ``Trainer`` calls the method instance directly so ``forward`` must be defined.
    def forward(self, batch):
        return self.step(batch)

    def step(self, batch):
        x = batch["x"].to(self.weight.device)
        loss = (self.weight * x).mean()
        return {"loss": loss}

    def on_train_start(self) -> None:
        raise RuntimeError("method hook boom")


class _RaisingFidelityProbe:
    def maybe_log(self, **_kwargs) -> None:
        raise RuntimeError("fidelity boom")


def _make_loader(num_batches: int = 1):
    class _Loader:
        def __iter__(self):
            for _ in range(num_batches):
                yield {"x": torch.ones(4, 1)}

        def __len__(self):
            return num_batches

    return _Loader()


def _make_trainer(**kwargs) -> Trainer:
    method = _MethodWithHooks()
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
    return Trainer(
        method,
        optimizer,
        console=ConsoleMonitor(),
        **kwargs,
    )


def test_trainer_logs_method_on_train_start(caplog, tmp_path: Path) -> None:
    trainer = _make_trainer(save_dir=str(tmp_path / "train_start"))

    with caplog.at_level(logging.WARNING):
        trainer.fit(_make_loader(1), epochs=1)

    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("method.on_train_start" in msg and "epoch=1" in msg for msg in messages)


def test_trainer_logs_fidelity_probe_exception(caplog, tmp_path: Path) -> None:
    trainer = _make_trainer(
        save_dir=str(tmp_path / "fidelity"),
        fidelity_probe=_RaisingFidelityProbe(),
    )

    with caplog.at_level(logging.WARNING):
        trainer.fit(_make_loader(1), epochs=1)

    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any(
        "fidelity_probe.maybe_log" in msg and "epoch=1" in msg and "step=1" in msg
        for msg in messages
    )
