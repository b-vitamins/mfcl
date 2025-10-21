"""Distributed communication logging utilities."""

from __future__ import annotations

import csv
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
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
class _OpAccumulator:
    all_reduce: float = 0.0
    all_gather: float = 0.0
    reduce_scatter: float = 0.0
    broadcast: float = 0.0

    def reset(self) -> None:
        for item in fields(self):
            setattr(self, item.name, 0.0)

    def add(self, kind: str, value: float) -> bool:
        if not hasattr(self, kind):
            return False
        current = getattr(self, kind)
        setattr(self, kind, current + value)
        return True

    def total(self) -> float:
        return sum(getattr(self, item.name) for item in fields(self))


_CATEGORY_FIELD_SPECS: dict[PayloadCategory, tuple[str, str]] = {
    PayloadCategory.FEATURES_ALLGATHER: ("features_allgather", "bytes_features_allgather"),
    PayloadCategory.MOMENTS_MU: ("moments_mu", "bytes_moments_mu"),
    PayloadCategory.MOMENTS_SIGMA_FULL: ("moments_sigma_full", "bytes_moments_sigma_full"),
    PayloadCategory.MOMENTS_SIGMA_DIAG: ("moments_sigma_diag", "bytes_moments_sigma_diag"),
    PayloadCategory.MIXTURE_MU_K: ("mixture_muK", "bytes_mixture_muK"),
    PayloadCategory.MIXTURE_SIGMA_K: ("mixture_sigmaK", "bytes_mixture_sigmaK"),
    PayloadCategory.THIRD_MOMENT_SKETCH: ("third_moment_sketch", "bytes_third_moment_sketch"),
    PayloadCategory.TOPR_INDICES: ("topr_indices", "bytes_topr_indices"),
    PayloadCategory.OTHER: ("other", "bytes_other"),
}


@dataclass
class _CategoryAccumulator:
    features_allgather: float = 0.0
    moments_mu: float = 0.0
    moments_sigma_full: float = 0.0
    moments_sigma_diag: float = 0.0
    mixture_muK: float = 0.0
    mixture_sigmaK: float = 0.0
    third_moment_sketch: float = 0.0
    topr_indices: float = 0.0
    other: float = 0.0

    def reset(self) -> None:
        for item in fields(self):
            setattr(self, item.name, 0.0)

    def add(self, category: PayloadCategory, value: float) -> bool:
        spec = _CATEGORY_FIELD_SPECS.get(category)
        if spec is None:
            return False
        attr_name, _ = spec
        current = getattr(self, attr_name)
        setattr(self, attr_name, current + value)
        return True

    def total(self) -> float:
        return sum(getattr(self, item.name) for item in fields(self))


@dataclass
class _StepAccumulators:
    epoch: int = 0
    global_step: int = 0
    world_size: int = 1
    timer: Optional[object] = None
    op_bytes: _OpAccumulator = field(default_factory=_OpAccumulator)
    op_time_s: _OpAccumulator = field(default_factory=_OpAccumulator)
    category_bytes: _CategoryAccumulator = field(default_factory=_CategoryAccumulator)

    def reset(self) -> None:
        self.op_bytes.reset()
        self.op_time_s.reset()
        self.category_bytes.reset()


class CommsLogger:
    """Append communication telemetry to ``comms.csv``."""

    def __init__(self, *, log_path: Path | None) -> None:
        self._log_path = Path(log_path) if log_path is not None else None
        self._csv_handle: IO[str] | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._active = False
        self._step = _StepAccumulators()
        self._last_step_totals: dict[str, float] | None = None
        self._step.reset()

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
        added = self._step.op_bytes.add(kind, bytes_on_wire)
        if not added:
            warnings.warn(
                f"Encountered unrecognized collective kind '{kind}'; event ignored.",
                RuntimeWarning,
            )
            return

        self._step.op_time_s.add(kind, max(0.0, duration_s))
        if not self._step.category_bytes.add(category, bytes_on_wire):
            warnings.warn(
                f"Encountered unrecognized payload category '{category}'; bytes not recorded.",
                RuntimeWarning,
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
        bytes_total = self._step.op_bytes.total()
        total_time_s = self._step.op_time_s.total()
        row = {
            "step": self._step.global_step,
            "epoch": self._step.epoch,
            "world_size": self._step.world_size,
        }
        for kind in _OP_KINDS:
            row[f"bytes_{kind}"] = getattr(self._step.op_bytes, kind)
        row["bytes_total"] = bytes_total
        for kind in _OP_KINDS:
            row[f"t_{kind}_ms"] = getattr(self._step.op_time_s, kind) * 1000.0
        row["t_total_ms"] = total_time_s * 1000.0
        if total_time_s > 0:
            # Binary (MiB/s) and decimal (MB/s) throughput for clarity.
            row["eff_bandwidth_MiBps"] = (bytes_total / (1024 ** 2)) / total_time_s
            row["eff_bandwidth_MBps"] = (bytes_total / 1_000_000.0) / total_time_s
        else:
            row["eff_bandwidth_MiBps"] = 0.0
            row["eff_bandwidth_MBps"] = 0.0

        missing_categories: list[PayloadCategory] = []
        for category in PayloadCategory:
            spec = _CATEGORY_FIELD_SPECS.get(category)
            if spec is None:
                missing_categories.append(category)
                continue
            attr_name, row_key = spec
            row[row_key] = getattr(self._step.category_bytes, attr_name)
        if missing_categories:
            names = ", ".join(category.name for category in missing_categories)
            warnings.warn(
                "Missing CSV columns for payload categories: " + names,
                RuntimeWarning,
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
