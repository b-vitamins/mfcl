"""Top-K hardness telemetry for negative similarity monitoring."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence

import torch

_ACTIVE_MONITOR: "HardnessMonitor | None" = None


def get_active_monitor() -> "HardnessMonitor | None":
    """Return the monitor active for the current training step, if any."""

    return _ACTIVE_MONITOR


def _set_active_monitor(monitor: "HardnessMonitor | None") -> None:
    global _ACTIVE_MONITOR
    _ACTIVE_MONITOR = monitor


class HardnessMonitor:
    """Track top-K negative similarity quantiles and write to CSV."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: str | Path | None,
        is_main: bool = True,
        max_anchors: int = 512,
        max_negatives: int | None = 4096,
        topk: Sequence[int] = (1, 5),
        seed: int = 0,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self.seed = int(seed)
        self.enabled = bool(enabled and is_main)
        self.max_anchors = max(1, int(max_anchors))
        self.max_negatives = None if max_negatives is None else max(1, int(max_negatives))
        processed_topk = sorted({int(k) for k in topk if int(k) > 0})
        if not processed_topk:
            processed_topk = [1]
        self.topk: tuple[int, ...] = tuple(processed_topk)
        self._columns: List[str] = ["step", "epoch", "anchors_seen", "anchors_sampled"]
        for k in self.topk:
            self._columns.append(f"top{k}_p50")
            self._columns.append(f"top{k}_p90")
        self._csv_handle: IO[str] | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._active = False
        self._current_step: int = 0
        self._current_epoch: int = 0
        self._anchors_seen: int = 0
        self._reservoir_total: int = 0
        self._reservoir_count: int = 0
        self._reservoir: torch.Tensor | None = None
        self._rng = random.Random(self.seed)
        self._torch_gen = torch.Generator()
        self._torch_gen.manual_seed(self.seed)

    def close(self) -> None:
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            finally:
                self._csv_handle = None
                self._csv_writer = None

    def begin_step(self, *, epoch: int, step: int) -> None:
        if not self.enabled:
            return
        combined_seed = self.seed + epoch * 131071 + step
        self._rng.seed(combined_seed)
        self._torch_gen.manual_seed(combined_seed)
        self._active = True
        self._current_epoch = int(epoch)
        self._current_step = int(step)
        self._anchors_seen = 0
        self._reservoir_total = 0
        self._reservoir_count = 0
        self._reservoir = None
        _set_active_monitor(self)

    def add_negatives(self, negatives: torch.Tensor) -> None:
        if not self._active or not self.enabled:
            return
        if negatives.ndim == 1:
            negatives = negatives.unsqueeze(0)
        elif negatives.ndim > 2:
            negatives = negatives.view(negatives.shape[0], -1)
        self._anchors_seen += negatives.shape[0]
        if negatives.numel() == 0:
            return
        neg = negatives.detach().to(torch.float32)
        if self.max_negatives is not None and neg.shape[1] > self.max_negatives:
            cols = torch.randperm(neg.shape[1], generator=self._torch_gen)[: self.max_negatives]
            neg = neg.index_select(1, cols)
        if self._reservoir is None or self._reservoir.shape[1] != neg.shape[1]:
            self._reservoir = torch.empty((self.max_anchors, neg.shape[1]), dtype=torch.float32)
            self._reservoir_count = 0
            self._reservoir_total = 0
        flat = neg.cpu()
        for row in flat:
            self._reservoir_total += 1
            if self._reservoir_count < self.max_anchors:
                self._reservoir[self._reservoir_count].copy_(row)
                self._reservoir_count += 1
            else:
                j = self._rng.randint(0, self._reservoir_total - 1)
                if j < self.max_anchors:
                    self._reservoir[j].copy_(row)

    def end_step(self) -> None:
        if not self._active or not self.enabled:
            _set_active_monitor(None)
            self._active = False
            return
        _set_active_monitor(None)
        self._active = False
        if self._reservoir is None or self._reservoir_count == 0:
            return
        data = self._reservoir[: self._reservoir_count]
        metrics = self._compute_quantiles(data)
        row = {
            "step": int(self._current_step),
            "epoch": int(self._current_epoch),
            "anchors_seen": int(self._anchors_seen),
            "anchors_sampled": int(self._reservoir_count),
        }
        row.update(metrics)
        self._write_row(row)
        self._reservoir = None
        self._reservoir_count = 0
        self._reservoir_total = 0

    def _compute_quantiles(self, values: torch.Tensor) -> Dict[str, float]:
        results: Dict[str, float] = {}
        if values.numel() == 0:
            for k in self.topk:
                results[f"top{k}_p50"] = math.nan
                results[f"top{k}_p90"] = math.nan
            return results
        percentiles = torch.tensor([0.5, 0.9], dtype=values.dtype)
        for k in self.topk:
            eff_k = min(k, values.shape[1])
            if eff_k <= 0:
                results[f"top{k}_p50"] = math.nan
                results[f"top{k}_p90"] = math.nan
                continue
            top_vals = torch.topk(values, k=eff_k, dim=1).values[:, eff_k - 1]
            if top_vals.numel() == 0:
                results[f"top{k}_p50"] = math.nan
                results[f"top{k}_p90"] = math.nan
                continue
            quantiles = torch.quantile(top_vals, percentiles.to(top_vals.device))
            results[f"top{k}_p50"] = float(quantiles[0].item())
            results[f"top{k}_p90"] = float(quantiles[1].item())
        return results

    def _write_row(self, row: Dict[str, float]) -> None:
        if self.log_path is None:
            return
        if self._csv_handle is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.log_path.exists()
            self._csv_handle = self.log_path.open("a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self._columns)
            if not exists or self.log_path.stat().st_size == 0:
                self._csv_writer.writeheader()
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)
        self._csv_handle.flush()


__all__ = ["HardnessMonitor", "get_active_monitor"]
