from pathlib import Path

import torch

from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class DummyMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(8 * 8 * 3, 1)

    def step(self, batch):
        x = batch["view1"].flatten(1)
        y = self.net(x).mean()
        return {"loss": y}

    def on_optimizer_step(self):
        return


def _dummy_loader(B: int = 4, H: int = 8):
    class _It:
        def __iter__(self):
            for _ in range(5):
                yield {
                    "view1": torch.randn(B, 3, H, H),
                    "view2": torch.randn(B, 3, H, H),
                    "index": torch.arange(B),
                }

        def __len__(self):
            return 5

    return _It()


def test_trainer_smoke(tmp_path: Path):
    method = DummyMethod()
    opt = torch.optim.SGD(method.parameters(), lr=0.1)
    trainer = Trainer(method, opt, console=ConsoleMonitor(), save_dir=str(tmp_path))
    loader = _dummy_loader()
    trainer.fit(loader, epochs=1)
    # checkpoint should exist
    files = list(tmp_path.glob("ckpt_ep*.pt"))
    assert files, "expected checkpoint file"
