from typing import List

from torch.utils.data import Dataset

from mfcl.data.samplers import ClassPackedSampler


class _ToyDataset(Dataset[int]):
    def __init__(self, per_class: List[int]) -> None:
        self.targets: List[int] = []
        for cls, count in enumerate(per_class):
            self.targets.extend([cls] * count)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.targets)

    def __getitem__(self, index: int) -> int:  # type: ignore[override]
        return index


def _batch(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []


def test_class_packed_sampler_balanced_batches():
    ds = _ToyDataset([5, 5, 5, 5])
    sampler = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        seed=0,
        shuffle=False,
    )
    indices = list(iter(sampler))
    assert len(indices) == 16
    batch_size = 4
    for batch in _batch(indices, batch_size):
        classes = [ds.targets[idx] for idx in batch]
        assert len(set(classes)) == 2
        for cls in set(classes):
            assert classes.count(cls) == 2


def test_class_packed_sampler_epoch_shuffles():
    ds = _ToyDataset([6, 6, 6])
    sampler = ClassPackedSampler(
        ds,
        num_classes_per_batch=3,
        instances_per_class=1,
        seed=7,
        shuffle=True,
    )
    first = list(iter(sampler))
    sampler.set_epoch(5)
    second = list(iter(sampler))
    assert first != second


def test_class_packed_sampler_distributed_split():
    ds = _ToyDataset([6, 6, 6, 6])
    single = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=1,
        shuffle=False,
    )
    full_order = list(iter(single))

    rank0 = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=1,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    rank1 = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=1,
        shuffle=False,
        num_replicas=2,
        rank=1,
    )

    idx0 = list(iter(rank0))
    idx1 = list(iter(rank1))
    assert len(idx0) == len(idx1)
    assert set(idx0).isdisjoint(idx1)
    combined = sorted(idx0 + idx1)
    assert combined == sorted(full_order)


def test_class_packed_sampler_drop_last_behavior():
    ds = _ToyDataset([4, 4, 2])
    # drop_last=True should trim the final partial class combination
    drop_sampler = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        shuffle=False,
        drop_last=True,
    )
    drop_indices = list(iter(drop_sampler))
    assert drop_indices == [0, 1, 4, 5, 8, 9, 2, 3]

    keep_sampler = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        shuffle=False,
        drop_last=False,
    )
    keep_indices = list(iter(keep_sampler))
    assert keep_indices == [0, 1, 4, 5, 8, 9, 2, 3, 6, 7]


def test_class_packed_sampler_distributed_partial_batch():
    ds = _ToyDataset([4, 4, 2])
    base_sampler = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        shuffle=False,
        drop_last=False,
    )
    full_indices = list(iter(base_sampler))

    rank0 = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        shuffle=False,
        drop_last=False,
        num_replicas=2,
        rank=0,
    )
    rank1 = ClassPackedSampler(
        ds,
        num_classes_per_batch=2,
        instances_per_class=2,
        shuffle=False,
        drop_last=False,
        num_replicas=2,
        rank=1,
    )

    idx0 = list(iter(rank0))
    idx1 = list(iter(rank1))
    assert len(idx0) == 8
    assert len(idx1) == 2
    assert set(idx0).isdisjoint(idx1)
    combined = sorted(idx0 + idx1)
    assert combined == sorted(full_indices)
