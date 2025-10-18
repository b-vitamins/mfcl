from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from mfcl.core.factory import build_data, build_optimizer, build_sched
from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import cleanup, get_rank, get_world_size, init_distributed, unwrap_ddp


class _ToyDDPMethod(torch.nn.Module):
    def __init__(self, img_size: int) -> None:
        super().__init__()
        flat = 3 * img_size * img_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
        )
        self.head = torch.nn.Linear(16, 4)
        self.indices_seen: List[int] = []

    def step(self, batch):
        self.indices_seen.extend(batch["index"].detach().cpu().tolist())
        v1 = batch["view1"].to(next(self.parameters()).device)
        v2 = batch["view2"].to(next(self.parameters()).device)
        z1 = self.head(self.encoder(v1))
        z2 = self.head(self.encoder(v2))
        loss = torch.nn.functional.mse_loss(z1, z2.detach())
        loss = loss + torch.nn.functional.mse_loss(z2, z1.detach())
        return {"loss": loss}

    def forward(self, batch):
        return self.step(batch)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, tmpdir: str, epochs: int = 1) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    try:
        init_distributed(backend="gloo")

        cfg = Config(
            data=DataConfig(
                root="/tmp",
                name="synthetic",
                batch_size=4,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                persistent_workers=False,
                synthetic_train_size=8,
                synthetic_val_size=4,
            ),
            aug=AugConfig(img_size=64, local_crops=0),
            model=ModelConfig(),
            method=MethodConfig(name="simclr"),
            optim=OptimConfig(lr=0.05),
            train=TrainConfig(
                epochs=epochs,
                warmup_epochs=0,
                cosine=False,
                amp=False,
                save_dir=str(tmpdir),
                scheduler_step_on="epoch",
                log_interval=1,
                seed=7,
            ),
        )

        train_loader, _ = build_data(cfg)
        base_method = _ToyDDPMethod(cfg.aug.img_size)
        method = base_method.to(torch.device("cpu"))
        if get_world_size() > 1:
            method = torch.nn.parallel.DistributedDataParallel(method)

        sampler = getattr(train_loader, "sampler", None)
        assert sampler is not None
        sampler.set_epoch(1)
        local_indices = list(iter(sampler))

        optimizer = build_optimizer(cfg, method)
        scheduler = build_sched(cfg, optimizer, steps_per_epoch=len(train_loader))

        trainer = Trainer(
            method,
            optimizer,
            scheduler=scheduler,
            console=ConsoleMonitor(),
            device=torch.device("cpu"),
            save_dir=str(tmpdir),
            scheduler_step_on="epoch",
        )
        trainer.fit(train_loader, epochs=epochs, save_every=1, eval_every=1)

        assert getattr(sampler, "epoch", None) == epochs

        gathered: List[List[int]] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(gathered, [int(i) for i in local_indices])
        if get_rank() == 0:
            flat = sorted({idx for bucket in gathered for idx in bucket})
            assert flat == list(range(cfg.data.synthetic_train_size))

        method_ref = unwrap_ddp(method)
        assert isinstance(method_ref, _ToyDDPMethod)

        local_state = {
            k: v.detach().cpu().clone()
            for k, v in method_ref.state_dict().items()
        }
        gathered_states: List[dict] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(gathered_states, local_state)
        if get_rank() == 0:
            base_state = gathered_states[0]
            for replica in gathered_states[1:]:
                assert replica.keys() == base_state.keys()
                for key in base_state:
                    assert torch.allclose(base_state[key], replica[key])

        ckpt = Path(tmpdir) / "ckpt_ep0001.pt"
        latest = Path(tmpdir) / "latest.pt"
        if get_rank() == 0:
            assert ckpt.exists()
            assert latest.exists()
        else:
            assert ckpt.exists()

    finally:
        cleanup()


@pytest.mark.dist
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_training_two_procs(tmp_path: Path):
    world_size = 2
    port = _free_port()
    mp.spawn(
        _worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
