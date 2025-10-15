from torch.utils.data import Dataset

from mfcl.data.sampler import RepeatAugSampler, DistSamplerWrapper


class DummyDS(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_repeat_aug_sampler_repeats_and_length():
    ds = DummyDS()
    s = RepeatAugSampler(ds, repeats=2, shuffle=True, seed=1, drop_last=True)
    idxs = list(iter(s))
    assert len(idxs) == (len(ds) // 2) * 2 * 2  # base_len * repeats


def test_dist_sampler_wrapper_single_process():
    ds = DummyDS()
    s = DistSamplerWrapper(ds, shuffle=True, seed=1)
    idxs = list(iter(s))
    assert len(idxs) == len(ds)
