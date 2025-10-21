"""Utilities for overlapping host-to-device transfers with compute.

The module exposes :class:`PrefetchLoader`, a thin wrapper around iterable
data loaders that moves batches to CUDA devices on dedicated streams. It also
contains helper utilities used to recursively place arbitrary nested
structures on a device.
"""

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
    """Move ``obj`` to ``device`` while preserving container structure.

    Args:
        obj: Object, potentially nested, containing tensors to be moved.
        device: Destination CUDA device.
        channels_last: Whether 4D floating tensors should be converted to
            channels-last layout after transfer.

    Returns:
        The input ``obj`` with every contained tensor moved to ``device``.

    Raises:
        RuntimeError: Propagated if moving tensors to the device fails.
    """
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
    """Wrap an iterable and prefetch batches onto a CUDA device.

    Args:
        loader: Source iterable that yields CPU batches.
        device: CUDA device batches should be moved onto.
        channels_last: Whether transferred 4D floating tensors should be
            converted to channels-last memory format.
        prefetch_depth: Number of CUDA streams used for overlapping data
            transfers with compute.

    Returns:
        ``None``.

    Raises:
        RuntimeError: If CUDA is unavailable when instantiated.
    """

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
        self._has_iterated = False

    def __enter__(self) -> "PrefetchLoader":
        """Return ``self`` so instances can be used as context managers.

        Returns:
            ``self``.
        """

        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Synchronize streams and release queued batches on exit.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback object, if any.

        Returns:
            ``None``.
        """

        self.close()

    def __len__(self) -> int:
        """Return the length of the underlying loader.

        Returns:
            Number of batches the wrapped ``loader`` contains.
        """

        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> "PrefetchLoader":
        """Initialise CUDA streams and begin prefetching.

        Returns:
            ``self`` so the loader can be iterated multiple times.
        """

        self.close()
        self._iterator = iter(self.loader)
        self._has_iterated = True
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
        """Return the next prefetched batch.

        Returns:
            Batch yielded by the wrapped loader, moved to ``device``.

        Raises:
            RuntimeError: If ``__iter__`` has not been called.
            StopIteration: When the wrapped loader is exhausted.
        """

        if not self._queue:
            if self._iterator is None:
                if not self._has_iterated:
                    raise RuntimeError("PrefetchLoader was not iterated")
                self.close()
                raise StopIteration
            if not self._preload():
                self.close()
                raise StopIteration
        stream, batch = self._queue.popleft()
        torch.cuda.current_stream(self.device).wait_stream(stream)  # type: ignore[arg-type]
        has_next = self._preload()
        if not has_next and not self._queue:
            self.close()
        return batch

    def close(self) -> None:
        """Synchronize streams and release references to prefetched batches.

        Returns:
            ``None``.
        """

        while self._queue:
            stream, _ = self._queue.popleft()
            stream.synchronize()
        self._queue.clear()
        for stream in self._streams:
            stream.synchronize()
        self._streams.clear()
        self._iterator = None
        self._next_stream_idx = 0

    def _preload(self) -> bool:
        """Fetch the next batch and enqueue it for future consumption.

        Returns:
            ``True`` if a batch was queued, ``False`` when the loader is
            exhausted.
        """

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

