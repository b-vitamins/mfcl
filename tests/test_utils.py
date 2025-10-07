import time
import torch

from mfcl.utils.dist import DistEnv
from mfcl.utils.timing import cuda_timing, get_peak_memory_gb, reset_peak_memory


def test_distenv_bytes_singleprocess():
    env = DistEnv()
    env.init()
    x = torch.randn(8, 4)
    y = env.all_gather_concat(x)
    assert torch.allclose(x, y)
    env.all_reduce(x.sum())
    stats = env.get_bytes().as_dict()
    assert stats["bytes_all_gather_payload"] == 0
    assert stats["bytes_all_reduce_payload"] == 0


def test_cuda_timer_cpu_path_runs():
    with cuda_timing(torch.device("cpu")) as t:
        time.sleep(0.01)
    assert t.elapsed_ms() > 0.5
    reset_peak_memory(torch.device("cpu"))
    assert get_peak_memory_gb(torch.device("cpu")) == 0.0
