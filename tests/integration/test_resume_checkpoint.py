from pathlib import Path

import torch

from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class DummyMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, batch):
        return self.step(batch)

    def step(self, batch):
        return {"loss": self.fc(batch["x"]).mean()}

    def on_optimizer_step(self):
        return


def _loader(N=2):
    class _It:
        def __iter__(self):
            for _ in range(N):
                yield {"x": torch.randn(8, 4)}

        def __len__(self):
            return N

    return _It()


def test_resume_checkpoint(tmp_path: Path):
    # Train 1 epoch and save
    m1 = DummyMethod()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    t1 = Trainer(m1, opt1, console=ConsoleMonitor(), save_dir=str(tmp_path / "r"))
    t1.fit(_loader(), epochs=1)
    ckpt = next((tmp_path / "r").glob("ckpt_ep0001.pt"))

    # Resume and run 1 more epoch
    m2 = DummyMethod()
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
    t2 = Trainer(m2, opt2, console=ConsoleMonitor(), save_dir=str(tmp_path / "r2"))
    t2.fit(_loader(), epochs=2, resume_path=str(ckpt))
