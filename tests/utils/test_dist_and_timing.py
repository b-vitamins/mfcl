import math
import time

import pytest
import torch

from mfcl.utils.dist import ByteCounters, DistEnv
from mfcl.utils.timing import CudaTimer, cuda_timing, get_peak_memory_gb, reset_peak_memory


def test_distenv_singleprocess_allreduce_and_gather() -> None:
    env = DistEnv()
    env.init()
    x = torch.arange(6, dtype=torch.float32).view(3, 2)

    gathered = env.all_gather_concat(x)
    assert torch.allclose(gathered, x)

    reduced = env.all_reduce(x.sum().clone())
    assert torch.allclose(reduced, x.sum())

    stats = env.get_bytes().as_dict()
    assert stats["bytes_all_gather_payload"] == 0
    assert stats["bytes_all_reduce_payload"] == 0

    env.bytes = ByteCounters(all_reduce_payload_bytes=10)
    env.reset_bytes()
    assert env.get_bytes().as_dict() == {
        "bytes_all_reduce_payload": 0,
        "bytes_all_reduce_theoretical": 0,
        "bytes_all_gather_payload": 0,
        "bytes_all_gather_theoretical": 0,
    }


def test_cudatimer_manual_usage_and_errors() -> None:
    timer = CudaTimer(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        timer.elapsed_ms()

    timer.start()
    time.sleep(0.01)
    elapsed = timer.stop()
    assert elapsed > 0.5
    assert math.isclose(elapsed, timer.elapsed_ms(), rel_tol=1e-6)


def test_cuda_timing_context_manager_cpu() -> None:
    with cuda_timing(torch.device("cpu")) as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms() > 0.5


def test_peak_memory_helpers_cpu_are_noops() -> None:
    reset_peak_memory(torch.device("cpu"))
    assert get_peak_memory_gb(torch.device("cpu")) == 0.0
