"""Class-balanced sampler ensuring packed batches by label."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset, Sampler


def _to_int_list(values: Iterable[int]) -> List[int]:
    result: List[int] = []
    for item in values:
        try:
            result.append(int(item))
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("Class targets must be convertible to int") from exc
    return result


def _extract_targets(dataset: Dataset[object]) -> List[int]:
    """Best-effort target extraction supporting common torchvision datasets."""

    if hasattr(dataset, "targets"):
        raw = getattr(dataset, "targets")
        if torch.is_tensor(raw):
            return _to_int_list(raw.tolist())
        if isinstance(raw, (list, tuple)):
            return _to_int_list(raw)
        return _to_int_list(list(raw))  # type: ignore[arg-type]
    if hasattr(dataset, "labels"):
        raw = getattr(dataset, "labels")
        if torch.is_tensor(raw):
            return _to_int_list(raw.tolist())
        if isinstance(raw, (list, tuple)):
            return _to_int_list(raw)
        return _to_int_list(list(raw))  # type: ignore[arg-type]
    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        if isinstance(samples, Sequence):
            labels: List[int] = []
            for sample in samples:
                if isinstance(sample, Sequence) and len(sample) >= 2:
                    labels.append(int(sample[1]))
            if labels:
                return labels
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = _extract_targets(getattr(dataset, "dataset"))
        indices = getattr(dataset, "indices")
        return [base_targets[int(i)] for i in indices]
    raise TypeError(
        "ClassPackedSampler requires the dataset to expose 'targets', 'labels',"
        " or 'samples' with class indices."
    )


class ClassPackedSampler(Sampler[int]):
    """Sampler that yields batches with a fixed class composition."""

    def __init__(
        self,
        dataset: Dataset[object],
        *,
        num_classes_per_batch: int,
        instances_per_class: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if num_classes_per_batch <= 0:
            raise ValueError("num_classes_per_batch must be > 0")
        if instances_per_class <= 0:
            raise ValueError("instances_per_class must be > 0")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not (0 <= rank < num_replicas):
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.num_classes_per_batch = int(num_classes_per_batch)
        self.instances_per_class = int(instances_per_class)
        self.batch_size = self.num_classes_per_batch * self.instances_per_class
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        targets = _extract_targets(dataset)
        if len(targets) != len(dataset):
            raise ValueError("Number of targets does not match dataset length")
        class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, cls in enumerate(targets):
            class_to_indices[int(cls)].append(idx)
        if not class_to_indices:
            raise ValueError("ClassPackedSampler found no class labels to sample")
        if len(class_to_indices) < self.num_classes_per_batch:
            raise ValueError(
                "ClassPackedSampler requires at least num_classes_per_batch distinct classes "
                f"(found={len(class_to_indices)}, "
                f"num_classes_per_batch={self.num_classes_per_batch})."
            )

        min_count = min(len(indices) for indices in class_to_indices.values())
        if min_count < self.instances_per_class:
            raise ValueError(
                "ClassPackedSampler requires at least instances_per_class samples per class "
                f"(min found={min_count}, instances_per_class={self.instances_per_class})."
            )
        self._class_to_indices = class_to_indices
        self._plan_epoch = -1
        self._global_batches: List[List[int]] = []
        self._local_batches: List[List[int]] = []
        self._generator = torch.Generator()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        self._ensure_plan()
        return len(self._local_batches) * self.batch_size

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        self._ensure_plan()
        for batch in self._local_batches:
            for index in batch:
                yield index

    def _ensure_plan(self) -> None:
        if self._plan_epoch == self.epoch:
            return
        self._generator.manual_seed(self.seed + self.epoch)
        self._global_batches = self._build_batches(generator=self._generator)
        self._local_batches = self._partition_batches(self._global_batches)
        self._plan_epoch = self.epoch

    def _build_batches(self, *, generator: torch.Generator) -> List[List[int]]:
        class_bins: List[deque[int]] = []
        per_class_sequences: Dict[int, List[int]] = {}
        for cls, indices in self._class_to_indices.items():
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator)
                shuffled = [indices[int(i)] for i in perm.tolist()]
            else:
                shuffled = list(indices)
            groups = len(shuffled) // self.instances_per_class
            if groups == 0:
                continue
            per_class_sequences[cls] = shuffled
            class_bins.append(deque([cls] * groups))

        if not class_bins:
            return []
        if self.shuffle and len(class_bins) > 1:
            perm_cls = torch.randperm(len(class_bins), generator=generator)
            class_bins = [class_bins[int(i)] for i in perm_cls.tolist()]

        class_slots: List[int] = []
        while any(class_bins):
            for bin_list in class_bins:
                if not bin_list:
                    continue
                class_slots.append(bin_list.popleft())

        total_batches = len(class_slots) // self.num_classes_per_batch
        if total_batches == 0:
            return []
        usable = total_batches * self.num_classes_per_batch
        class_slots = class_slots[:usable]

        class_offsets: Dict[int, int] = {cls: 0 for cls in per_class_sequences}
        batches: List[List[int]] = []
        for batch_idx in range(total_batches):
            start = batch_idx * self.num_classes_per_batch
            end = start + self.num_classes_per_batch
            selected = class_slots[start:end]
            batch: List[int] = []
            for cls in selected:
                seq = per_class_sequences[cls]
                offset = class_offsets[cls]
                span = seq[offset : offset + self.instances_per_class]
                if len(span) < self.instances_per_class:
                    continue
                batch.extend(span)
                class_offsets[cls] = offset + self.instances_per_class
            if len(batch) == self.batch_size:
                batches.append(batch)
        return batches

    def _partition_batches(self, batches: List[List[int]]) -> List[List[int]]:
        if self.num_replicas == 1:
            return batches
        per_rank = len(batches) // self.num_replicas
        usable = per_rank * self.num_replicas
        if usable == 0:
            return []
        batches = batches[:usable]
        start = self.rank * per_rank
        end = start + per_rank
        return batches[start:end]


__all__ = ["ClassPackedSampler"]
