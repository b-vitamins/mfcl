"""Distributed communication logging utilities."""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import IO, Optional, Any

import torch
import torch.distributed as dist


class PayloadCategory(Enum):
    """Semantic tags for collective payloads.

    The communication logger uses these categories to attribute the number of
    bytes transferred by a collective to higher-level algorithmic concepts.
    Consumers of the telemetry file may rely on the following conventions:

    Attributes:
        FEATURES_ALLGATHER: Feature tensors gathered across all ranks prior to
            model evaluation or scoring.
        MOMENTS_MU: First-moment (mean) statistics shared between ranks.
        MOMENTS_SIGMA_FULL: Full covariance matrices communicated when fitting
            multivariate models.
        MOMENTS_SIGMA_DIAG: Diagonal covariance statistics broadcast or
            reduced across ranks.
        MIXTURE_MU_K: Mean parameters for mixture model components.
        MIXTURE_SIGMA_K: Covariance parameters for mixture model components.
        THIRD_MOMENT_SKETCH: Sketches of higher-order moment tensors used for
            moment-matching algorithms.
        TOPR_INDICES: Indices exchanged for top-*r* selection routines.
        OTHER: Any payload that does not fall into one of the specialised
            categories above.
    """

    FEATURES_ALLGATHER = auto()
    MOMENTS_MU = auto()
    MOMENTS_SIGMA_FULL = auto()
    MOMENTS_SIGMA_DIAG = auto()
    MIXTURE_MU_K = auto()
    MIXTURE_SIGMA_K = auto()
    THIRD_MOMENT_SKETCH = auto()
    TOPR_INDICES = auto()
    OTHER = auto()


_OP_KINDS = ("all_reduce", "all_gather", "reduce_scatter", "broadcast")


@dataclass
class _StepAccumulators:
    epoch: int = 0
    global_step: int = 0
    world_size: int = 1
    timer: Optional[object] = None
    op_bytes: dict[str, float] | None = None
    op_time_s: dict[str, float] | None = None
    category_bytes: dict[PayloadCategory, float] | None = None

    def reset(self) -> None:
        self.op_bytes = {kind: 0.0 for kind in _OP_KINDS}
        self.op_time_s = {kind: 0.0 for kind in _OP_KINDS}
        self.category_bytes = {category: 0.0 for category in PayloadCategory}


class CommsLogger:
    """Append communication telemetry to ``comms.csv``."""

    def __init__(self, *, log_path: Path | None) -> None:
        self._log_path = Path(log_path) if log_path is not None else None
        self._csv_handle: IO[str] | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._active = False
        self._step = _StepAccumulators()
        self._last_step_totals: dict[str, float] | None = None

    @property
    def enabled(self) -> bool:
        return self._log_path is not None

    def begin_step(
        self,
        *,
        epoch: int,
        step_index: int,
        global_step: int,
        timer: Optional[object],
    ) -> None:
        del step_index  # unused but kept for parity with StepTimer
        self._active = True
        self._step.reset()
        self._step.epoch = int(epoch)
        self._step.global_step = int(global_step)
        self._step.world_size = _safe_world_size()
        # Store the per-step timer so we can attribute communication latency.
        # The StepTimer exposes ``add_comm_ms``; keeping a reference avoids using
        # globals and keeps the coupling explicit for future contributors.
        self._step.timer = timer

    def record_event(
        self,
        *,
        kind: str,
        tensor: torch.Tensor,
        category: PayloadCategory,
        duration_s: float,
    ) -> None:
        if not self._active:
            return
        if not torch.is_tensor(tensor):  # pragma: no cover - defensive
            return
        size_bytes = float(tensor.element_size() * tensor.numel())
        bytes_on_wire = _estimate_wire_bytes(kind, size_bytes, self._step.world_size)
        if self._step.op_bytes is None or self._step.op_time_s is None:
            self._step.reset()
        assert self._step.op_bytes is not None
        assert self._step.op_time_s is not None
        assert self._step.category_bytes is not None
        self._step.op_bytes[kind] = self._step.op_bytes.get(kind, 0.0) + bytes_on_wire
        self._step.op_time_s[kind] = self._step.op_time_s.get(kind, 0.0) + max(0.0, duration_s)
        self._step.category_bytes[category] = (
            self._step.category_bytes.get(category, 0.0) + bytes_on_wire
        )

        timer = self._step.timer
        if timer is not None and hasattr(timer, "add_comm_ms"):
            try:
                timer.add_comm_ms(duration_s * 1000.0)
            except Exception:  # pragma: no cover - defensive
                pass

    def end_step(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._step.op_bytes is None or self._step.op_time_s is None:
            return
        bytes_total = sum(self._step.op_bytes.get(kind, 0.0) for kind in _OP_KINDS)
        total_time_s = sum(self._step.op_time_s.get(kind, 0.0) for kind in _OP_KINDS)
        row = {
            "step": self._step.global_step,
            "epoch": self._step.epoch,
            "world_size": self._step.world_size,
        }
        for kind in _OP_KINDS:
            row[f"bytes_{kind}"] = self._step.op_bytes.get(kind, 0.0)
        row["bytes_total"] = bytes_total
        for kind in _OP_KINDS:
            row[f"t_{kind}_ms"] = self._step.op_time_s.get(kind, 0.0) * 1000.0
        row["t_total_ms"] = total_time_s * 1000.0
        if total_time_s > 0:
            # Binary (MiB/s) and decimal (MB/s) throughput for clarity.
            row["eff_bandwidth_MiBps"] = (bytes_total / (1024 ** 2)) / total_time_s
            row["eff_bandwidth_MBps"] = (bytes_total / 1_000_000.0) / total_time_s
        else:
            row["eff_bandwidth_MiBps"] = 0.0
            row["eff_bandwidth_MBps"] = 0.0

        assert self._step.category_bytes is not None
        row.update(
            {
                "bytes_features_allgather": self._step.category_bytes[PayloadCategory.FEATURES_ALLGATHER],
                "bytes_moments_mu": self._step.category_bytes[PayloadCategory.MOMENTS_MU],
                "bytes_moments_sigma_full": self._step.category_bytes[PayloadCategory.MOMENTS_SIGMA_FULL],
                "bytes_moments_sigma_diag": self._step.category_bytes[PayloadCategory.MOMENTS_SIGMA_DIAG],
                "bytes_mixture_muK": self._step.category_bytes[PayloadCategory.MIXTURE_MU_K],
                "bytes_mixture_sigmaK": self._step.category_bytes[PayloadCategory.MIXTURE_SIGMA_K],
                "bytes_third_moment_sketch": self._step.category_bytes[PayloadCategory.THIRD_MOMENT_SKETCH],
                "bytes_topr_indices": self._step.category_bytes[PayloadCategory.TOPR_INDICES],
                "bytes_other": self._step.category_bytes[PayloadCategory.OTHER],
            }
        )

        self._last_step_totals = {"bytes_total": bytes_total}
        self._write_row(row)

    def close(self) -> None:
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None

    def pop_last_step_totals(self) -> dict[str, float] | None:
        """Return the totals from the previously completed step."""

        totals = self._last_step_totals
        self._last_step_totals = None
        return totals

    def _write_row(self, row: dict[str, float]) -> None:
        if self._log_path is None:
            return
        if self._csv_writer is None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_handle = self._log_path.open("a", newline="")
            fieldnames = [
                "step",
                "epoch",
                "world_size",
                "bytes_all_reduce",
                "bytes_all_gather",
                "bytes_reduce_scatter",
                "bytes_broadcast",
                "bytes_total",
                "t_all_reduce_ms",
                "t_all_gather_ms",
                "t_reduce_scatter_ms",
                "t_broadcast_ms",
                "t_total_ms",
                "eff_bandwidth_MiBps",
                "eff_bandwidth_MBps",
                "bytes_features_allgather",
                "bytes_moments_mu",
                "bytes_moments_sigma_full",
                "bytes_moments_sigma_diag",
                "bytes_mixture_muK",
                "bytes_mixture_sigmaK",
                "bytes_third_moment_sketch",
                "bytes_topr_indices",
                "bytes_other",
            ]
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=fieldnames)
            if self._log_path.stat().st_size == 0:
                self._csv_writer.writeheader()
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)
        if self._csv_handle is not None:
            self._csv_handle.flush()


_ACTIVE_LOGGER: CommsLogger | None = None


def configure_comms_logger(
    *, enabled: bool, log_path: Path | None, is_main: bool
) -> CommsLogger | None:
    global _ACTIVE_LOGGER
    if _ACTIVE_LOGGER is not None:
        _ACTIVE_LOGGER.close()
        _ACTIVE_LOGGER = None
    if not enabled or not is_main:
        return None
    logger = CommsLogger(log_path=log_path)
    _ACTIVE_LOGGER = logger
    return logger


def get_comms_logger() -> CommsLogger | None:
    return _ACTIVE_LOGGER


def close_comms_logger() -> None:
    global _ACTIVE_LOGGER
    if _ACTIVE_LOGGER is not None:
        _ACTIVE_LOGGER.close()
        _ACTIVE_LOGGER = None


class AsyncCollectiveHandle:
    """Wrap an asynchronous work handle to emit telemetry.

    The wrapper intercepts lifecycle calls such as :meth:`wait` and
    :meth:`is_completed` to ensure that the communication logger records the
    duration of an asynchronous collective exactly once.

    Args:
        kind: Name of the collective (for example ``"all_reduce"``).
        tensor: Representative payload tensor whose size determines the number
            of bytes transferred on the wire.
        category: Semantic category assigned to the payload.
        work: Underlying asynchronous work handle returned by
            :mod:`torch.distributed` or a compatible stub used for testing.
        logger: Active communication logger.
        start_time_s: Wall-clock timestamp captured immediately before the
            collective was issued.
    """

    def __init__(
        self,
        *,
        kind: str,
        tensor: torch.Tensor,
        category: PayloadCategory,
        work: Any,
        logger: "CommsLogger",
        start_time_s: float | None = None,
    ) -> None:
        self._kind = kind
        self._tensor = tensor
        self._category = category
        self._work = work
        self._logger = logger
        self._start_time_s = start_time_s if start_time_s is not None else time.perf_counter()
        self._logged = False

    def wait(self, *args, **kwargs):  # type: ignore[override]
        """Block until the work completes and record the telemetry event."""

        try:
            return self._work.wait(*args, **kwargs)
        finally:
            self._finalize()

    def is_completed(self) -> bool:  # type: ignore[override]
        """Return whether the work has finished and log the event if so."""

        completed = self._work.is_completed()
        if completed:
            self._finalize()
        return completed

    def synchronize(self):  # type: ignore[override]
        """Synchronise with the completion of the work and log the event."""

        if hasattr(self._work, "synchronize"):
            try:
                return self._work.synchronize()
            finally:
                self._finalize()
        return self.wait()

    def result(self):  # type: ignore[override]
        """Proxy to :meth:`torch.distributed.Work.result` with logging."""

        if hasattr(self._work, "result"):
            try:
                return self._work.result()
            finally:
                self._finalize()
        return self.wait()

    def exception(self):  # type: ignore[override]
        """Return the underlying exception, logging the event beforehand."""

        self._finalize()
        if hasattr(self._work, "exception"):
            return self._work.exception()
        return None

    def _finalize(self) -> None:
        if self._logged:
            return
        duration = time.perf_counter() - self._start_time_s
        self._logger.record_event(
            kind=self._kind, tensor=self._tensor, category=self._category, duration_s=duration
        )
        self._logged = True

    def __getattr__(self, name: str):
        return getattr(self._work, name)

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            if hasattr(self._work, "is_completed") and self._work.is_completed():
                self._finalize()
        except Exception:
            pass


def is_logging_enabled() -> bool:
    """Return ``True`` when collective telemetry should be recorded."""

    logger = get_comms_logger()
    if logger is None:
        return False
    return _safe_world_size() > 1


@contextmanager
def log_collective(kind: str, tensor: torch.Tensor, category: PayloadCategory):
    logger = get_comms_logger()
    if logger is None or not is_logging_enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.record_event(kind=kind, tensor=tensor, category=category, duration_s=duration)


def _safe_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_world_size()
        except Exception:  # pragma: no cover - defensive
            return 1
    return 1


def _estimate_wire_bytes(kind: str, size_bytes: float, world_size: int) -> float:
    if world_size <= 1:
        return 0.0
    if kind == "all_reduce":
        return 2.0 * (world_size - 1) / world_size * size_bytes
    if kind in {"all_gather", "reduce_scatter"}:
        return (world_size - 1) * size_bytes
    if kind == "broadcast":
        return size_bytes
    return size_bytes


__all__ = [
    "PayloadCategory",
    "CommsLogger",
    "AsyncCollectiveHandle",
    "configure_comms_logger",
    "get_comms_logger",
    "close_comms_logger",
    "is_logging_enabled",
    "log_collective",
]
