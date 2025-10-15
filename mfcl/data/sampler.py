"""Samplers: repeat-augmentation and distributed wrapper."""

from __future__ import annotations

from typing import Iterator, Optional, Any, cast
from collections.abc import Sized

import torch
from torch.utils.data import Dataset, Sampler


class RepeatAugSampler(Sampler[int]):
    """Sampler that repeats each index 'repeats' times per epoch with reshuffling.

    Useful when each sample produces random multi-view augmentations and we want
    more uniform coverage across workers. Repeated indices are shuffled after
    repetition to reduce local duplicate clustering.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        repeats: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        """Construct RepeatAugSampler.

        Args:
            dataset: Target dataset.
            repeats: How many times to repeat indices per epoch (>=1).
            shuffle: If True, shuffle base indices each epoch.
            seed: Base seed for deterministic shuffling.
            drop_last: Whether to drop tail indices to make num_samples divisible by repeats.

        Raises:
            ValueError: If repeats < 1.
        """
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        # torch Dataset typing doesn't declare Sized; compute length once.
        self.dataset = dataset
        self._n = len(cast(Sized, dataset))
        self.repeats = int(repeats)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch to alter shuffling deterministically across epochs."""

        self.epoch = int(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        n = self._n
        base = n - (n % self.repeats) if self.drop_last else n
        return base * self.repeats

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        n = self._n
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.arange(n)
        if self.shuffle:
            indices = indices[torch.randperm(n, generator=g)]
        if self.drop_last:
            base_len = (n // self.repeats) * self.repeats
            indices = indices[:base_len]
        # Repeat each index 'repeats' times
        idxs = indices.repeat_interleave(self.repeats)
        if self.shuffle:
            perm = torch.randperm(idxs.numel(), generator=g)
            idxs = idxs[perm]
        yield from (int(i) for i in idxs)


class DistSamplerWrapper(Sampler[int]):
    """Minimal wrapper over torch.utils.data.distributed.DistributedSampler.

    Provides identical API in single-process mode (degenerate sampler).
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        from torch.utils.data.distributed import DistributedSampler
        import torch.distributed as dist

        self.dataset = dataset
        self._n = len(cast(Sized, dataset))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._dist_sampler: Optional[DistributedSampler] = None
        if dist.is_available() and dist.is_initialized():
            self._dist_sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self._dist_sampler is not None:
            self._dist_sampler.set_epoch(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        if self._dist_sampler is not None:
            return len(self._dist_sampler)
        if self.drop_last:
            return self._n  # drop_last has no practical effect without replication
        return self._n

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        if self._dist_sampler is not None:
            yield from iter(self._dist_sampler)
            return
        n = self._n
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            for i in torch.randperm(n, generator=g):
                yield int(i)
            return
        yield from range(n)


__all__ = ["RepeatAugSampler", "DistSamplerWrapper"]
