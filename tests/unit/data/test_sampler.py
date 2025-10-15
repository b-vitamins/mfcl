from collections import Counter

from torch.utils.data import Dataset

from mfcl.data.sampler import RepeatAugSampler, DistSamplerWrapper


class DummyDS(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_repeat_aug_sampler_repeats_and_scatter():
    ds = DummyDS()
    repeats = 2
    s = RepeatAugSampler(ds, repeats=repeats, shuffle=True, seed=1, drop_last=True)
    idxs = list(iter(s))
    base_len = (len(ds) // repeats) * repeats
    assert len(idxs) == base_len * repeats
    counts = Counter(idxs)
    assert all(v == repeats for v in counts.values())
    assert any(idxs[i] != idxs[i + 1] for i in range(len(idxs) - 1))


def test_repeat_aug_sampler_set_epoch_changes_order():
    ds = DummyDS()
    s = RepeatAugSampler(ds, repeats=2, shuffle=True, seed=1, drop_last=False)
    first = list(iter(s))
    s.set_epoch(1)
    second = list(iter(s))
    assert first != second


def test_dist_sampler_wrapper_single_process():
    ds = DummyDS()
    s = DistSamplerWrapper(ds, shuffle=True, seed=1)
    idxs = list(iter(s))
    assert len(idxs) == len(ds)
    assert len(s) == len(ds)
