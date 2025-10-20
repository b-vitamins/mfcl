import os
import socket
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mfcl.engines.trainer import Trainer
from mfcl.runtime.beta_ctrl import BetaController
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import cleanup, init_distributed


class _DummyMethod(torch.nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.mixture_beta_raw = beta
        self.mixture_beta = beta

    def forward(self, batch):  # pragma: no cover - not used in test
        return {"loss": self.weight.sum()}

    def step(self, batch):  # pragma: no cover - not used in test
        return {"loss": self.weight.sum()}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    try:
        init_distributed(backend="gloo")
        method = _DummyMethod(beta=6.0 + rank)
        optimizer = torch.optim.SGD(method.parameters(), lr=0.01)
        controller = BetaController(
            target_eps=0.05,
            beta_min=3.0,
            beta_max=12.0,
            ema_window=4,
            is_main=(rank == 0),
        )
        stats = {
            "pi": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "pi_min": torch.tensor(0.5, dtype=torch.float32),
            "median_xBx": torch.tensor(1.0e-4, dtype=torch.float32),
            "delta_sigma_max": torch.tensor(0.0, dtype=torch.float32),
        }
        estimator = type("_StubEstimator", (), {"_last_stats": stats})()
        trainer = Trainer(
            method,
            optimizer,
            console=ConsoleMonitor(),
            device=torch.device("cpu"),
            beta_controller=controller,
            mixture_estimator=estimator,
        )

        trainer._maybe_update_beta_controller(epoch=0, global_step=1)

        applied = float(getattr(trainer._method_impl, "mixture_beta"))
        gathered: List[float] = [0.0] * world_size
        dist.all_gather_object(gathered, applied)
        if rank == 0:
            base = gathered[0]
            for value in gathered[1:]:
                assert abs(value - base) < 1e-6
    finally:
        cleanup()


@pytest.mark.dist
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_beta_controller_broadcasts_beta_across_ranks():
    world_size = 2
    port = _free_port()
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
