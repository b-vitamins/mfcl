"""Utilities for overlapping host-to-device transfers with compute."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import torch


def _move_to_device(
    obj: Any,
    device: torch.device,
    *,
    channels_last: bool,
) -> Any:
    if torch.is_tensor(obj):
        tensor = obj.to(device=device, non_blocking=True)
        if (
            channels_last
            and tensor.ndim == 4
            and tensor.is_floating_point()
        ):
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, channels_last=channels_last) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_move_to_device(v, device, channels_last=channels_last) for v in obj]
        if isinstance(obj, tuple):
            try:
                if hasattr(obj, "_fields"):
                    return type(obj)(*seq)
                return type(obj)(seq)
            except TypeError:
                try:
                    return type(obj)(*seq)
                except TypeError:
                    return tuple(seq)
        return seq
    return obj


class PrefetchLoader(Iterable[Any]):
    """Wrap a DataLoader to prefetch batches onto a CUDA device."""

    def __init__(
        self,
        loader: Iterable[Any],
        device: torch.device,
        *,
        channels_last: bool = False,
        prefetch_depth: int = 1,
    ) -> None:
        if not torch.cuda.is_available():  # pragma: no cover - defensive
            raise RuntimeError("PrefetchLoader requires CUDA availability")
        self.loader = loader
        self.device = device
        self.channels_last = bool(channels_last)
        self.prefetch_depth = max(1, int(prefetch_depth))
        self._iterator: Optional[Iterator[Any]] = None
        self._streams: list[torch.cuda.Stream] = []
        self._queue: deque[tuple[torch.cuda.Stream, Any]] = deque()
        self._next_stream_idx = 0

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> "PrefetchLoader":
        self._iterator = iter(self.loader)
        self._streams = [
            torch.cuda.Stream(device=self.device) for _ in range(self.prefetch_depth)
        ]
        self._queue.clear()
        self._next_stream_idx = 0
        for _ in range(self.prefetch_depth):
            if not self._preload():
                break
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            raise RuntimeError("PrefetchLoader was not iterated")
        if not self._queue:
            raise StopIteration
        stream, batch = self._queue.popleft()
        torch.cuda.current_stream(self.device).wait_stream(stream)  # type: ignore[arg-type]
        self._preload()
        return batch

    def _preload(self) -> bool:
        if self._iterator is None:
            return False
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            return False

        stream = self._streams[self._next_stream_idx]
        self._next_stream_idx = (self._next_stream_idx + 1) % len(self._streams)
        with torch.cuda.stream(stream):
            moved = _move_to_device(
                batch,
                self.device,
                channels_last=self.channels_last,
            )
        self._queue.append((stream, moved))
        return True


__all__ = ["PrefetchLoader"]

