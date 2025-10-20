"""Distributed communication logging utilities."""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import IO, Optional

import torch
import torch.distributed as dist


class PayloadCategory(Enum):
    """Semantic tags for collective payloads."""

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

    @property
    def enabled(self) -> bool:
        return True

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
            # Report binary-prefixed throughput to match the 1024 scaling.
            row["eff_bandwidth_MiBps"] = (bytes_total / (1024 ** 2)) / total_time_s
        else:
            row["eff_bandwidth_MiBps"] = 0.0

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

        self._write_row(row)

    def close(self) -> None:
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None

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


@contextmanager
def log_collective(kind: str, tensor: torch.Tensor, category: PayloadCategory):
    logger = get_comms_logger()
    if logger is None:
        yield
        return
    if _safe_world_size() <= 1:
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
    "configure_comms_logger",
    "get_comms_logger",
    "close_comms_logger",
    "log_collective",
]
