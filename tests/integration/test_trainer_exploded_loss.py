import pytest
import torch

from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class NaNMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Provide a dummy parameter so optimizers can be constructed
        self._dummy = torch.nn.Parameter(torch.tensor(0.0))

    def step(self, batch):
        return {"loss": torch.tensor(float("nan"))}

    def on_optimizer_step(self):
        return


def _loader_once():
    class _It:
        def __iter__(self):
            yield {"x": torch.tensor(1.0)}

        def __len__(self):
            return 1

    return _It()


def test_trainer_raises_on_nan_loss(tmp_path):
    m = NaNMethod()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    t = Trainer(m, opt, console=ConsoleMonitor(), save_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        t.fit(_loader_once(), epochs=1)
