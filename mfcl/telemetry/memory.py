"""GPU memory telemetry utilities."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import IO, Dict, Optional

import torch

try:  # pragma: no cover - optional dependency
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - NVML is optional at runtime
    pynvml = None  # type: ignore


_BYTES_PER_MB = 1024.0 * 1024.0


class MemoryMonitor:
    """Periodically record GPU memory statistics.

    Two data sources are combined:

    * A background sampler thread that queries NVML at 1 Hz for process-wide
      device memory usage.
    * Explicit per-step snapshots collected from ``torch.cuda.memory_stats``
      every ``step_interval`` steps.

    When NVML or CUDA is unavailable the monitor silently disables itself.
    """

    _CSV_COLUMNS = [
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
    ]

    def __init__(
        self,
        *,
        enabled: bool,
        step_interval: int,
        log_path: str | Path | None,
        is_main: bool,
        device_index: Optional[int] = None,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self.enabled = bool(enabled and is_main and self.log_path is not None)
        self._step_interval = max(1, int(step_interval))
        self._device_index = device_index
        self._csv_handle: Optional[IO[str]] = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._nvml_initialized = False
        self._last_epoch = 0
        self._last_step = 0
        self._last_global_step = 0
        self._breakdown: Dict[str, float] = {
            "mixture_buffers_MB": 0.0,
            "resp_buffer_MB": 0.0,
            "sketch_buffer_MB": 0.0,
        }
        self._cuda_available = torch.cuda.is_available()
        if self._device_index is None and self._cuda_available:
            try:
                self._device_index = torch.cuda.current_device()
            except Exception:
                self._device_index = 0

        if not self.enabled or not self._cuda_available:
            return

        self._open_csv()
        self._init_nvml()
        if self._nvml_handle is not None:
            self._sampler = threading.Thread(
                target=self._sampler_loop, name="nvml-memory-sampler", daemon=True
            )
            self._sampler.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush buffers and stop the sampler thread."""

        if not self.enabled:
            return
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
            self._sampler = None
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None
        if self._nvml_initialized and pynvml is not None:
            try:  # pragma: no cover - best effort shutdown
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def update_step_context(
        self, *, epoch: int, step_index: int, global_step: int
    ) -> None:
        """Update the epoch/step context for sampler annotations."""

        if not self.enabled:
            return
        self._last_epoch = int(epoch)
        self._last_step = int(step_index)
        self._last_global_step = int(global_step)

    def record_step_snapshot(self, *, epoch: int, global_step: int) -> None:
        """Record a snapshot of ``torch.cuda.memory_stats`` for the step."""

        if not self.enabled or not self._cuda_available:
            return
        if global_step <= 0 or (global_step % self._step_interval) != 0:
            return
        stats: Dict[str, int] = {}
        try:
            device = self._device_index if self._device_index is not None else 0
            raw_stats = torch.cuda.memory_stats(device)
            if isinstance(raw_stats, dict):
                stats = {str(k): int(v) for k, v in raw_stats.items()}
        except Exception:
            stats = {}
        row = self._base_row(
            epoch=epoch,
            step=global_step,
            event="step",
        )
        row["details"] = json.dumps(stats, sort_keys=True)
        self._log_row(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_csv(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        self._csv_handle = self.log_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self._CSV_COLUMNS)
        if need_header:
            self._csv_writer.writeheader()
            self._csv_handle.flush()

    def _init_nvml(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            device = self._device_index if self._device_index is not None else 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:
            self._nvml_handle = None
            if self._nvml_initialized:
                try:  # pragma: no cover - defensive
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._nvml_initialized = False

    def _collect_torch_metrics(self) -> tuple[float, float, float]:
        if not self._cuda_available:
            return 0.0, 0.0, 0.0
        device = self._device_index if self._device_index is not None else 0
        try:
            used = torch.cuda.memory_allocated(device) / _BYTES_PER_MB
        except Exception:
            used = 0.0
        try:
            reserved = torch.cuda.memory_reserved(device) / _BYTES_PER_MB
        except Exception:
            reserved = 0.0
        try:
            max_alloc = torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB
        except Exception:
            max_alloc = 0.0
        return used, reserved, max_alloc

    def _base_row(
        self,
        *,
        epoch: int,
        step: int,
        event: str,
        used_override: Optional[float] = None,
    ) -> Dict[str, object]:
        used_mb, reserved_mb, max_alloc_mb = self._collect_torch_metrics()
        if used_override is not None:
            used_mb = float(max(0.0, used_override))
        row: Dict[str, object] = {
            "timestamp": time.time(),
            "event": event,
            "step": int(step),
            "epoch": int(epoch),
            "mem_used_MB": float(max(0.0, used_mb)),
            "mem_reserved_MB": float(max(0.0, reserved_mb)),
            "mem_max_allocated_MB": float(max(0.0, max_alloc_mb)),
            "mixture_buffers_MB": float(self._breakdown["mixture_buffers_MB"]),
            "resp_buffer_MB": float(self._breakdown["resp_buffer_MB"]),
            "sketch_buffer_MB": float(self._breakdown["sketch_buffer_MB"]),
            "details": "",
        }
        return row

    def _log_row(self, row: Dict[str, object]) -> None:
        if self._csv_writer is None or self._csv_handle is None:
            return
        with self._lock:
            self._csv_writer.writerow(row)
            self._csv_handle.flush()

    def _sampler_loop(self) -> None:
        assert self._nvml_handle is not None
        while not self._stop_event.is_set():
            used_mb = 0.0
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                used_mb = float(getattr(info, "used", 0)) / _BYTES_PER_MB
            except Exception:
                used_mb = 0.0
            row = self._base_row(
                epoch=self._last_epoch,
                step=self._last_global_step,
                event="nvml",
                used_override=used_mb,
            )
            self._log_row(row)
            self._stop_event.wait(1.0)


__all__ = ["MemoryMonitor"]
