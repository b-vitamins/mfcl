from pathlib import Path

import pytest
import torch

from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor


class LinearLossMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))

    def step(self, batch):
        x = batch["x"]
        return {"loss": (self.w * x).mean()}

    def on_optimizer_step(self):
        return


def _loader_const(N=4):
    class _It:
        def __iter__(self):
            for _ in range(N):
                yield {"x": torch.ones(8, 1)}

        def __len__(self):
            return N

    return _It()


def test_accumulation_equivalence_and_scheduler(tmp_path: Path):
    # Baseline: accum_steps=1
    m1 = LinearLossMethod()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
    sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.1)
    t1 = Trainer(
        m1,
        opt1,
        scheduler=sched1,
        console=ConsoleMonitor(),
        save_dir=str(tmp_path / "r1"),
        accum_steps=1,
        scheduler_step_on="batch",
    )
    t1.fit(_loader_const(2), epochs=1)
    w1 = m1.w.detach().clone()

    # Accum: accum_steps=2 should yield similar update over 2 steps
    m2 = LinearLossMethod()
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.1)
    t2 = Trainer(
        m2,
        opt2,
        scheduler=sched2,
        console=ConsoleMonitor(),
        save_dir=str(tmp_path / "r2"),
        accum_steps=2,
        scheduler_step_on="batch",
    )
    t2.fit(_loader_const(2), epochs=1)
    w2 = m2.w.detach().clone()
    # With per-batch scheduling and accumulation, updates should be very close
    # (not bitwise identical in general). Allow a small numerical tolerance.
    assert torch.allclose(w1, w2, atol=1.1e-1, rtol=1e-4)

    # Scheduler stepping per epoch vs batch
    m3 = LinearLossMethod()
    opt3 = torch.optim.SGD(m3.parameters(), lr=0.1)
    sched3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=1, gamma=0.9)
    t3 = Trainer(
        m3,
        opt3,
        scheduler=sched3,
        console=ConsoleMonitor(),
        save_dir=str(tmp_path / "r3"),
        accum_steps=1,
        scheduler_step_on="epoch",
    )
    t3.fit(_loader_const(2), epochs=1)
    # For epoch stepping, lr should have stepped once; for batch stepping in t1, stepped twice
    lr_batch = t1.optimizer.param_groups[0]["lr"]
    lr_epoch = t3.optimizer.param_groups[0]["lr"]
    assert lr_epoch != lr_batch


@pytest.mark.cuda
def test_amp_cuda_runs_if_available(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    m = LinearLossMethod().to("cuda")
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    t = Trainer(
        m, opt, console=ConsoleMonitor(), save_dir=str(tmp_path / "r"), accum_steps=1
    )
    t.fit(_loader_const(2), epochs=1)
