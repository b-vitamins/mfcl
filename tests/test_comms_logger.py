import csv
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mfcl.distributed import all_gather, all_reduce, PayloadCategory
from mfcl.telemetry.comms_logger import (
    CommsLogger,
    close_comms_logger,
    configure_comms_logger,
)


def test_comms_logger_enabled_property(tmp_path):
    log_path = tmp_path / "comms.csv"
    logger_with_path = CommsLogger(log_path=log_path)
    assert logger_with_path.enabled is True

    logger_without_path = CommsLogger(log_path=None)
    assert logger_without_path.enabled is False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _setup_env(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)


def _worker_basic(rank: int, world_size: int, port: int, tmpdir: str) -> None:
    _setup_env(rank, world_size, port)
    dist.init_process_group("gloo")
    try:
        log_path = Path(tmpdir) / "comms.csv"
        logger = configure_comms_logger(
            enabled=True, log_path=log_path, is_main=(rank == 0)
        )
        if logger is not None:
            logger.begin_step(epoch=1, step_index=1, global_step=1, timer=None)

        tensor = torch.ones(4, dtype=torch.float32) * (rank + 1)
        all_reduce(tensor, category=PayloadCategory.MOMENTS_MU)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather(gathered, tensor, category=PayloadCategory.FEATURES_ALLGATHER)

        assert torch.allclose(tensor, torch.full_like(tensor, 3.0))
        expected = [torch.full_like(tensor, 3.0) for _ in range(world_size)]
        for idx, bucket in enumerate(gathered):
            assert torch.allclose(bucket, expected[idx])

        if logger is not None:
            logger.end_step()

        dist.barrier()

        if rank == 0:
            assert log_path.exists()
            with log_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1
            row = rows[0]
            bytes_all_reduce = float(row["bytes_all_reduce"])
            bytes_all_gather = float(row["bytes_all_gather"])
            bytes_total = float(row["bytes_total"])
            assert bytes_all_reduce == pytest.approx(16.0)
            assert bytes_all_gather == pytest.approx(16.0)
            assert bytes_total == pytest.approx(32.0)
            assert float(row["bytes_reduce_scatter"]) == pytest.approx(0.0)
            assert float(row["bytes_broadcast"]) == pytest.approx(0.0)
            total_ms = float(row["t_total_ms"])
            op_sum = (
                float(row["t_all_reduce_ms"]) + float(row["t_all_gather_ms"]) + float(row["t_reduce_scatter_ms"]) + float(row["t_broadcast_ms"])
            )
            assert total_ms == pytest.approx(op_sum)
            assert float(row["bytes_moments_mu"]) == pytest.approx(16.0)
            assert float(row["bytes_features_allgather"]) == pytest.approx(16.0)
            assert float(row["bytes_other"]) == pytest.approx(0.0)
            assert float(row["bytes_topr_indices"]) == pytest.approx(0.0)
            assert float(row["eff_bandwidth_MiBps"]) >= 0.0

            cat_sum = (
                float(row["bytes_features_allgather"])
                + float(row["bytes_moments_mu"])
                + float(row["bytes_moments_sigma_full"])
                + float(row["bytes_moments_sigma_diag"])
                + float(row["bytes_mixture_muK"])
                + float(row["bytes_mixture_sigmaK"])
                + float(row["bytes_third_moment_sketch"])
                + float(row["bytes_topr_indices"])
                + float(row["bytes_other"])
            )
            assert cat_sum == pytest.approx(bytes_total)
    finally:
        close_comms_logger()
        dist.destroy_process_group()


def _worker_disabled(rank: int, world_size: int, port: int, tmpdir: str) -> None:
    _setup_env(rank, world_size, port)
    dist.init_process_group("gloo")
    try:
        log_path = Path(tmpdir) / "comms.csv"
        logger = configure_comms_logger(
            enabled=False, log_path=log_path, is_main=(rank == 0)
        )
        assert logger is None

        tensor = torch.ones(4, dtype=torch.float32) * (rank + 1)
        all_reduce(tensor, category=PayloadCategory.OTHER)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather(gathered, tensor, category=PayloadCategory.OTHER)

        dist.barrier()

        if rank == 0:
            assert not log_path.exists()
    finally:
        close_comms_logger()
        dist.destroy_process_group()


@pytest.mark.dist
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_comms_logger_records_bytes(tmp_path):
    world_size = 2
    port = _free_port()
    mp.spawn(
        _worker_basic,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.dist
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_comms_logger_respects_feature_flag(tmp_path):
    world_size = 2
    port = _free_port()
    mp.spawn(
        _worker_disabled,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
