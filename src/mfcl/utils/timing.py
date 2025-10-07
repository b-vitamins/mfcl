from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

import torch


class CudaTimer:
    """
    CUDA wall-clock timer using CUDA events when available, else perf_counter.
    Use as a context manager or manual start/stop.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._t0: Optional[float] = None
        self._event_start: Optional[torch.cuda.Event] = None
        self._event_end: Optional[torch.cuda.Event] = None
        self._elapsed_ms: Optional[float] = None

    def start(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._event_start = torch.cuda.Event(enable_timing=True)
            self._event_end = torch.cuda.Event(enable_timing=True)
            self._event_start.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self.device.type == "cuda":
            assert self._event_start is not None and self._event_end is not None
            self._event_end.record()
            self._event_end.synchronize()
            self._elapsed_ms = float(self._event_start.elapsed_time(self._event_end))
        else:
            assert self._t0 is not None
            self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return self._elapsed_ms

    def elapsed_ms(self) -> float:
        if self._elapsed_ms is None:
            raise RuntimeError("Timer not stopped yet.")
        return self._elapsed_ms


@contextmanager
def cuda_timing(device: Optional[torch.device] = None):
    t = CudaTimer(device=device)
    t.start()
    try:
        yield t
    finally:
        t.stop()


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gb(device: Optional[torch.device] = None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        bytes_alloc = torch.cuda.max_memory_allocated(device)
        return float(bytes_alloc) / (1024**3)
    return 0.0
