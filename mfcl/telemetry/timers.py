"""Fine-grained timing helpers for training telemetry."""

from __future__ import annotations

import csv
import time
from bisect import bisect_left, insort
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import IO, Dict, List, Optional

import torch


class StepTimer:
    """Per-step timer aggregating CPU and CUDA timing data."""

    _CSV_COLUMNS = [
        "step",
        "epoch",
        "t_data_ms",
        "t_fwd_ms",
        "t_bwd_ms",
        "t_opt_ms",
        "t_comm_ms",
        "t_assign_ms",
        "t_topr_ms",
        "t_beta_ctrl_ms",
        "t_misc_ms",
        "t_step_ms",
        "ips_step",
        "outlier_flags",
    ]

    def __init__(
        self,
        *,
        enabled: bool,
        warmup_steps: int = 0,
        sample_rate: int = 1,
        log_path: str | Path | None,
        nvtx_enabled: bool = False,
        is_main: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self.enabled = bool(enabled and is_main and self.log_path is not None)
        self.warmup_steps = max(0, int(warmup_steps))
        self.sample_rate = max(1, int(sample_rate))
        self._step_counter = 0
        self._active = False
        self._current: Dict[str, float] = {}
        self._current_step_id: int = 0
        self._current_epoch: int = 0
        self._cuda_available = torch.cuda.is_available()
        self._nvtx_enabled = bool(
            nvtx_enabled and self._cuda_available and hasattr(torch.cuda, "nvtx")
        )
        if self._nvtx_enabled:
            try:
                self._nvtx_push = torch.cuda.nvtx.range_push
                self._nvtx_pop = torch.cuda.nvtx.range_pop
            except Exception:
                self._nvtx_enabled = False
        self._cuda_segments = {"forward", "backward", "optimizer"}
        self._cpu_segments = {"assign", "topR", "beta_ctrl", "misc"}
        self._active_cuda: Dict[str, bool] = {name: False for name in self._cuda_segments}
        if self._cuda_available:
            self._cuda_events: Dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {
                name: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for name in self._cuda_segments
            }
        else:
            self._cuda_events = {}
        self._csv_handle: Optional[IO[str]] = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        self._comm_ms: float = 0.0
        self._rows_since_flush = 0
        self._flush_interval = 10
        window = 20
        self._step_history = deque(maxlen=window)
        self._comm_history = deque(maxlen=window)
        self._step_sorted: List[float] = []
        self._comm_sorted: List[float] = []

    def close(self) -> None:
        """Close the underlying CSV handle if it is open."""

        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None
                self._rows_since_flush = 0

    def begin_step(self, *, epoch: int, step_index: int, global_step: Optional[int] = None) -> None:
        """Mark the start of a new step and apply warmup/sample filters."""

        self._step_counter += 1
        should_record = (
            self.enabled
            and self._step_counter > self.warmup_steps
            and ((self._step_counter - self.warmup_steps - 1) % self.sample_rate == 0)
        )
        self._active = should_record
        if not should_record:
            return
        self._current = {
            "data_s": 0.0,
            "assign_s": 0.0,
            "topR_s": 0.0,
            "beta_ctrl_s": 0.0,
            "misc_s": 0.0,
            "forward_ms": 0.0,
            "backward_ms": 0.0,
            "optimizer_ms": 0.0,
        }
        self._comm_ms = 0.0
        self._current_step_id = int(global_step) if global_step is not None else self._step_counter
        self._current_epoch = int(epoch)
        for key in self._active_cuda:
            self._active_cuda[key] = False

    def record_data(self, seconds: float) -> None:
        if self._active:
            self._current["data_s"] = self._current.get("data_s", 0.0) + float(max(0.0, seconds))

    def set_comm_ms(self, value: float) -> None:
        if self._active:
            self._comm_ms = float(max(0.0, value))

    def add_comm_ms(self, value: float) -> None:
        if self._active:
            self._comm_ms = float(max(0.0, self._comm_ms + value))

    def range_forward(self):
        return self._segment_range("forward")

    def range_backward(self):
        return self._segment_range("backward")

    def range_optimizer(self):
        return self._segment_range("optimizer")

    def range_assign(self):
        return self._segment_range("assign")

    def range_topr(self):
        return self._segment_range("topR")

    def range_beta_ctrl(self):
        return self._segment_range("beta_ctrl")

    def range_misc(self):
        return self._segment_range("misc")

    def end_step(self, *, step_time_s: float, ips: Optional[float] = None) -> None:
        if not self._active:
            return
        step_ms = max(0.0, float(step_time_s) * 1000.0)
        data_ms = max(0.0, self._current.get("data_s", 0.0) * 1000.0)
        assign_ms = max(0.0, self._current.get("assign_s", 0.0) * 1000.0)
        topr_ms = max(0.0, self._current.get("topR_s", 0.0) * 1000.0)
        beta_ms = max(0.0, self._current.get("beta_ctrl_s", 0.0) * 1000.0)
        misc_manual_ms = max(0.0, self._current.get("misc_s", 0.0) * 1000.0)

        if self._cuda_available:
            need_sync = any(self._active_cuda.values())
            if need_sync:
                torch.cuda.synchronize()
            fwd_ms = self._elapsed_cuda("forward")
            bwd_ms = self._elapsed_cuda("backward")
            opt_ms = self._elapsed_cuda("optimizer")
        else:
            fwd_ms = float(self._current.get("forward_ms", 0.0))
            bwd_ms = float(self._current.get("backward_ms", 0.0))
            opt_ms = float(self._current.get("optimizer_ms", 0.0))

        comm_ms = max(0.0, self._comm_ms)
        known_total = (
            data_ms
            + fwd_ms
            + bwd_ms
            + opt_ms
            + comm_ms
            + assign_ms
            + topr_ms
            + beta_ms
            + misc_manual_ms
        )
        residual = step_ms - known_total
        misc_ms = max(0.0, misc_manual_ms + residual)

        flags: list[str] = []
        med = self._rolling_median(self._step_sorted)
        if med > 0.0 and step_ms > 3.0 * med:
            flags.append("step")
        med_comm = self._rolling_median(self._comm_sorted)
        if med_comm > 0.0 and comm_ms > 3.0 * med_comm:
            flags.append("comm")

        self._push_history(self._step_history, self._step_sorted, step_ms)
        self._push_history(self._comm_history, self._comm_sorted, comm_ms)

        row = {
            "step": self._current_step_id,
            "epoch": self._current_epoch,
            "t_data_ms": data_ms,
            "t_fwd_ms": fwd_ms,
            "t_bwd_ms": bwd_ms,
            "t_opt_ms": opt_ms,
            "t_comm_ms": comm_ms,
            "t_assign_ms": assign_ms,
            "t_topr_ms": topr_ms,
            "t_beta_ctrl_ms": beta_ms,
            "t_misc_ms": misc_ms,
            "t_step_ms": step_ms,
            "ips_step": float(ips) if ips is not None else 0.0,
            "outlier_flags": ";".join(flags) if flags else "",
        }
        self._write_row(row)
        self._active = False
        self._current = {}

    @staticmethod
    def _push_history(queue: deque[float], sorted_list: List[float], value: float) -> None:
        if queue.maxlen is not None and len(queue) == queue.maxlen:
            oldest = queue.popleft()
            idx = bisect_left(sorted_list, oldest)
            if 0 <= idx < len(sorted_list):
                del sorted_list[idx]
        queue.append(value)
        insort(sorted_list, value)

    @staticmethod
    def _rolling_median(sorted_list: List[float]) -> float:
        n = len(sorted_list)
        if n < 5:
            return 0.0
        mid = n // 2
        if n % 2:
            return float(sorted_list[mid])
        return float((sorted_list[mid - 1] + sorted_list[mid]) * 0.5)

    def _segment_range(self, name: str):
        if not self._active:
            return nullcontext()
        if name in self._cuda_segments:
            return self._cuda_range(name)
        if name in self._cpu_segments:
            return self._cpu_range(name)
        return nullcontext()

    @contextmanager
    def _cuda_range(self, name: str):
        if not self._active:
            yield
            return
        use_cuda = self._cuda_available
        start_time = 0.0
        if self._nvtx_enabled:
            self._nvtx_push(name)
        if use_cuda:
            start, _ = self._cuda_events[name]
            start.record()
        else:
            start_time = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                _, end = self._cuda_events[name]
                end.record()
                self._active_cuda[name] = True
            else:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                key = f"{name}_ms"
                self._current[key] = float(self._current.get(key, 0.0) + elapsed)
            if self._nvtx_enabled:
                self._nvtx_pop()

    @contextmanager
    def _cpu_range(self, name: str):
        if not self._active:
            yield
            return
        if self._nvtx_enabled:
            self._nvtx_push(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            key = f"{name}_s"
            self._current[key] = float(self._current.get(key, 0.0) + elapsed)
            if self._nvtx_enabled:
                self._nvtx_pop()

    def _elapsed_cuda(self, name: str) -> float:
        if not self._cuda_available or not self._active_cuda.get(name, False):
            return 0.0
        start, end = self._cuda_events[name]
        try:
            return float(start.elapsed_time(end))
        except RuntimeError:
            return 0.0

    def _ensure_writer(self) -> Optional[csv.DictWriter[str]]:
        if not self.enabled or self.log_path is None:
            return None
        if self._csv_writer is not None:
            return self._csv_writer
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        need_header = True
        if self.log_path.exists():
            try:
                need_header = self.log_path.stat().st_size == 0
            except OSError:
                need_header = True
        self._csv_handle = self.log_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self._CSV_COLUMNS)
        if need_header:
            self._csv_writer.writeheader()
            self._csv_handle.flush()
        return self._csv_writer

    def _write_row(self, row: Dict[str, float | int]) -> None:
        writer = self._ensure_writer()
        if writer is None:
            return
        writer.writerow(row)
        if self._csv_handle is not None:
            self._rows_since_flush += 1
            if self._flush_interval <= 1 or self._rows_since_flush >= self._flush_interval:
                self._csv_handle.flush()
                self._rows_since_flush = 0

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort
        try:
            self.close()
        except Exception:
            pass


__all__ = ["StepTimer"]
