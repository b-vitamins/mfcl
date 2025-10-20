"""GPU power and energy telemetry utilities."""

from __future__ import annotations

import csv
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional, Sequence

import torch

try:  # pragma: no cover - optional dependency at runtime
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - NVML is optional at runtime
    pynvml = None  # type: ignore


_DEFAULT_SAMPLE_INTERVAL_S = 1.0


class PowerMonitor:
    """Periodically sample NVML power usage and integrate energy."""

    _CSV_COLUMNS = [
        "timestamp",
        "step",
        "epoch",
        "gpu_id",
        "power_W",
        "energy_Wh_cum",
        "energy_J_cum",
    ]

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: str | Path | None,
        is_main: bool,
        sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S,
        kwh_price_usd: float = 0.0,
        start_sampler: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self.enabled = bool(enabled and is_main and self.log_path is not None)
        self.sample_interval_s = max(0.5, float(sample_interval_s))
        self.kwh_price_usd = float(kwh_price_usd)

        self._csv_handle: Optional[IO[str]] = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self._nvml_initialized = False
        self._nvml_handles: list[object] = []
        self._device_indices: list[int] = []
        self._energy_wh: list[float] = []
        self._energy_j: list[float] = []
        self._last_sample_time: Optional[float] = None
        self._last_epoch = 0
        self._last_step = 0
        self._last_global_step = 0
        self._logger = logging.getLogger(__name__)
        self._power_error_logged = False

        self._cuda_available = torch.cuda.is_available()
        self._device_count = torch.cuda.device_count() if self._cuda_available else 0

        if not self.enabled or not self._cuda_available or self._device_count <= 0:
            self.enabled = False
            return

        self._open_csv()
        if not self.enabled:
            return

        self._init_nvml()
        if not self.enabled:
            self._close_csv()
            return

        if start_sampler:
            self._sampler = threading.Thread(
                target=self._sampler_loop,
                name="nvml-power-sampler",
                daemon=True,
            )
            self._sampler.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the sampler thread and close resources."""

        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
            self._sampler = None
        self._close_csv()
        if self._nvml_initialized and pynvml is not None:
            try:  # pragma: no cover - best effort shutdown
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
        self.enabled = False

    def update_step_context(
        self, *, epoch: int, step_index: int, global_step: int
    ) -> None:
        """Update the epoch/step context for future samples."""

        if not self.enabled:
            return
        with self._lock:
            self._last_epoch = int(epoch)
            self._last_step = int(step_index)
            self._last_global_step = int(global_step)

    def get_totals(self) -> tuple[float, float]:
        """Return total cumulative energy across GPUs in Wh and Joules."""

        if not self.enabled:
            return (0.0, 0.0)
        with self._lock:
            return (float(sum(self._energy_wh)), float(sum(self._energy_j)))

    def get_epoch_cost(self, energy_wh: float) -> float:
        """Return the estimated USD cost for the provided energy in Wh."""

        if self.kwh_price_usd <= 0:
            return 0.0
        return (energy_wh / 1000.0) * self.kwh_price_usd

    # Internal helper exposed for testing.
    def _process_sample(self, timestamp: float, dt: float, powers: Sequence[float]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._ingest_locked(timestamp, dt, list(powers))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_csv(self) -> None:
        if self.log_path is None:
            self.enabled = False
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.log_path.exists()
            self._csv_handle = self.log_path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_handle, fieldnames=self._CSV_COLUMNS
            )
            if not file_exists or self._csv_handle.tell() == 0:
                self._csv_writer.writeheader()
                self._csv_handle.flush()
        except Exception as exc:  # pragma: no cover - filesystem errors
            self._logger.warning(
                "Disabling energy telemetry; unable to open %s: %s", self.log_path, exc
            )
            self.enabled = False
            self._csv_writer = None
            if self._csv_handle is not None:
                try:
                    self._csv_handle.close()
                finally:
                    self._csv_handle = None

    def _close_csv(self) -> None:
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None

    def _init_nvml(self) -> None:
        if pynvml is None:
            self._logger.warning("NVML not available; disabling energy telemetry.")
            self.enabled = False
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as exc:
            self._logger.warning(
                "Failed to initialize NVML; disabling energy telemetry: %s", exc
            )
            self.enabled = False
            return

        handles: list[object] = []
        indices: list[int] = []
        for idx in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception as exc:
                self._logger.warning(
                    "Unable to access NVML handle for GPU %s: %s", idx, exc
                )
                continue
            handles.append(handle)
            indices.append(idx)

        if not handles:
            self._logger.warning(
                "No NVML GPU handles available; disabling energy telemetry."
            )
            self.enabled = False
            try:
                if self._nvml_initialized:
                    pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._nvml_initialized = False
            return

        self._nvml_handles = handles
        self._device_indices = indices
        self._energy_wh = [0.0 for _ in handles]
        self._energy_j = [0.0 for _ in handles]

    def _sampler_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            if self._stop_event.wait(self.sample_interval_s):
                break

    def _sample_once(self) -> None:
        if not self.enabled:
            return
        powers = self._read_powers()
        if powers is None:
            return
        timestamp = time.time()
        with self._lock:
            dt = 0.0
            if self._last_sample_time is not None:
                dt = max(0.0, timestamp - self._last_sample_time)
            self._last_sample_time = timestamp
            self._ingest_locked(timestamp, dt, powers)

    def _read_powers(self) -> Optional[list[float]]:
        if pynvml is None:
            return None
        powers: list[float] = []
        for handle in self._nvml_handles:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            except Exception as exc:
                if not self._power_error_logged:
                    self._logger.warning(
                        "NVML power query failed; disabling energy telemetry: %s",
                        exc,
                    )
                self._power_error_logged = True
                self.enabled = False
                self._stop_event.set()
                return None
            if power_mw is None:
                powers.append(0.0)
            else:
                powers.append(float(power_mw) / 1000.0)
        return powers

    def _ingest_locked(self, timestamp: float, dt: float, powers: Sequence[float]) -> None:
        if not self._csv_writer:
            return
        timestamp_str = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        for idx, power_w in enumerate(powers):
            if idx >= len(self._energy_wh):
                break
            energy_wh_inc = power_w * (dt / 3600.0)
            energy_j_inc = power_w * dt
            self._energy_wh[idx] += energy_wh_inc
            self._energy_j[idx] += energy_j_inc
            row = {
                "timestamp": timestamp_str,
                "step": self._last_global_step,
                "epoch": self._last_epoch,
                "gpu_id": self._device_indices[idx] if idx < len(self._device_indices) else idx,
                "power_W": power_w,
                "energy_Wh_cum": self._energy_wh[idx],
                "energy_J_cum": self._energy_j[idx],
            }
            self._csv_writer.writerow(row)
        if self._csv_handle is not None:
            self._csv_handle.flush()


__all__ = ["PowerMonitor"]
