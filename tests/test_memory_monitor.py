import csv
import json
import time
from pathlib import Path

import torch

from mfcl.telemetry.memory import MemoryMonitor


class _FakeNVML:
    class MemoryInfo:
        def __init__(self, used: int) -> None:
            self.used = used

    def __init__(self, used_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.initialized = False

    def nvmlInit(self) -> None:  # pragma: no cover - trivial
        self.initialized = True

    def nvmlShutdown(self) -> None:  # pragma: no cover - trivial
        self.initialized = False

    def nvmlDeviceGetHandleByIndex(self, index: int):  # pragma: no cover - trivial
        return index

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self.MemoryInfo(self.used_bytes)


def test_memory_monitor_logs_expected_columns(tmp_path, monkeypatch):
    log_path = Path(tmp_path) / "memory.csv"

    fake_nvml = _FakeNVML(used_bytes=256 * 1024 * 1024)
    monkeypatch.setattr("mfcl.telemetry.memory.pynvml", fake_nvml, raising=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=0: 128 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=0: 192 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=0: 320 * 1024 * 1024)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device=0: {"allocated_bytes.all.peak": 512 * 1024 * 1024},
    )

    monitor = MemoryMonitor(
        enabled=True,
        step_interval=1,
        log_path=log_path,
        is_main=True,
        device_index=0,
    )
    try:
        monitor.update_step_context(epoch=1, step_index=1, global_step=1)
        monitor.record_step_snapshot(epoch=1, global_step=1)
        time.sleep(1.2)
    finally:
        monitor.close()

    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "memory monitor did not emit any rows"
    header = rows[0].keys()
    expected = {
        "timestamp",
        "event",
        "step",
        "epoch",
        "mem_used_MB",
        "mem_reserved_MB",
        "mem_max_allocated_MB",
        "mixture_buffers_MB",
        "resp_buffer_MB",
        "sketch_buffer_MB",
        "details",
    }
    assert expected.issubset(set(header))

    step_rows = [row for row in rows if row["event"] == "step"]
    assert step_rows, "expected at least one step snapshot"
    details = json.loads(step_rows[0]["details"])
    assert "allocated_bytes.all.peak" in details

    nvml_rows = [row for row in rows if row["event"] == "nvml"]
    assert nvml_rows, "expected NVML sampler rows"
    used = float(nvml_rows[0]["mem_used_MB"])
    assert used > 0
