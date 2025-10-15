"""Simple meters for smoothed and cumulative averages."""

from __future__ import annotations

from collections import deque
from typing import Deque


class SmoothedValue:
    """Track a series of values with windowed and global averages."""

    def __init__(self, window: int = 20) -> None:
        self.window: int = int(window)
        self.deque: Deque[float] = deque(maxlen=self.window)
        self.total: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(float(value))
        self.count += int(n)
        self.total += float(value) * int(n)

    def reset(self) -> None:
        """Reset window and globals."""
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    @property
    def median(self) -> float:
        d = list(self.deque)
        if not d:
            return 0.0
        s = sorted(d)
        m = len(s)
        mid = m // 2
        if m % 2 == 0:
            return 0.5 * (s[mid - 1] + s[mid])
        return s[mid]

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0


class AverageMeter:
    """Cumulative average meter."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.n += int(n)

    @property
    def avg(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum / self.n


__all__ = ["SmoothedValue", "AverageMeter"]
