"""Console monitor for single-line live updates and epoch summaries.

Produces minimal, tidy logs suitable for terminals and text files. No color,
no progress bars, just concise single-line updates and summaries.
"""

from __future__ import annotations

import sys
import time
import shutil
from typing import Dict, Sequence


class ConsoleMonitor:
    """Single-line monitor with epoch summaries and ETA."""

    def __init__(self, time_smooth: float = 0.98):
        """Initialize the console monitor.

        Args:
            time_smooth: EMA factor for step time smoothing in [0,1).
        """
        if not (0.0 <= time_smooth < 1.0):
            raise ValueError("time_smooth must be in [0,1)")
        self.time_smooth = time_smooth
        self._last_ts = time.time()
        self._ema_step = 0.0
        self._last_len = 0

    def reset_epoch_timer(self) -> None:
        """Reset internal step-time EMA, e.g., call at epoch start."""

        self._ema_step = 0.0
        self._last_ts = time.time()

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def epoch_start(
        self,
        epoch: int,
        total: int | None = None,
        *,
        header: Sequence[str] | None = None,
    ) -> None:
        """Print a short epoch banner and reset timers."""

        self.newline()
        msg = f"Epoch {epoch:03d}"
        if total is not None and total > 0:
            msg += f" • batches={total}"
        print(msg)
        if header:
            print(" ".join(header))
        self.reset_epoch_timer()

    @staticmethod
    def _fmt_metric(name: str, value: float) -> str:
        """Format metrics with human-friendly precision based on their name."""

        if name in {"lr", "learning_rate"}:
            return f"{float(value):.6f}"
        if name in {"ips", "imgs_per_sec", "images_per_sec"}:
            return f"{float(value):.1f}"
        if "time" in name:
            return f"{float(value):.3f}"
        return f"{float(value):.4f}"

    def live(
        self,
        epoch: int,
        step: int,
        total: int,
        metrics: Dict[str, float],
        metric_order: Sequence[str] | None = None,
    ) -> None:
        """Render an updating line for batch metrics.

        Args:
            epoch: Current epoch index.
            step: Current batch index (1-based ideally).
            total: Total batches per epoch.
            metrics: Mapping of metric name to scalar float.
            metric_order: Optional explicit ordering for metrics when displaying.
        Notes:
            ETA renders as 00:00:00 when total or step are non-positive.
        """
        now = time.time()
        dt = max(1e-9, now - self._last_ts)
        self._last_ts = now
        if self._ema_step <= 0:
            self._ema_step = dt
        else:
            a = self.time_smooth
            self._ema_step = a * self._ema_step + (1.0 - a) * dt

        if total <= 0 or step <= 0:
            eta = 0.0
        else:
            remaining = max(0, total - step)
            eta = self._ema_step * remaining

        parts = [f"ep {epoch:03d} [{step:05d}/{total:05d}]"]
        seen: set[str] = set()
        if metric_order is not None:
            for name in metric_order:
                if name in metrics:
                    seen.add(name)
                    value = metrics[name]
                    try:
                        parts.append(f"{name}={self._fmt_metric(name, float(value))}")
                    except Exception:
                        continue
        for k, v in metrics.items():
            if k in seen:
                continue
            try:
                parts.append(f"{k}={self._fmt_metric(k, float(v))}")
            except Exception:
                # Skip non-floaty values
                continue
        parts.append(f"eta={self._fmt_eta(eta)}")
        msg = " ".join(parts)

        # Respect terminal width; truncate if needed.
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        if len(msg) > width - 1:
            msg = msg[: max(0, width - 1)]

        # Right-pad to clear residual characters from previous line.
        pad = max(0, self._last_len - len(msg))
        sys.stdout.write("\r" + msg + (" " * pad))
        sys.stdout.flush()
        self._last_len = len(msg)

    def newline(self) -> None:
        """Commit the current line and move to the next (prints a newline)."""
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_len = 0

    def summary(
        self,
        epoch: int,
        metrics: Dict[str, float],
        metric_order: Sequence[str] | None = None,
    ) -> None:
        """Print a single tidy summary line after an epoch.

        Example:
            "[epoch 005] loss=0.4321 top1=67.12 time=12.34m"

        Args:
            epoch: Completed epoch index.
            metrics: Mapping of metric name to scalar float.
            metric_order: Optional explicit ordering for metrics when displaying.
        """
        parts = [f"[epoch {epoch:03d}]"]
        seen: set[str] = set()
        if metric_order is not None:
            for name in metric_order:
                if name in metrics:
                    seen.add(name)
                    value = metrics[name]
                    try:
                        parts.append(f"{name}={self._fmt_metric(name, float(value))}")
                    except Exception:
                        continue
        for k, v in metrics.items():
            if k in seen:
                continue
            try:
                parts.append(f"{k}={self._fmt_metric(k, float(v))}")
            except Exception:
                continue
        msg = " ".join(parts)
        print(msg)


__all__ = ["ConsoleMonitor"]
