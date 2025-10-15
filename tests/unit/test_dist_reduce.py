import types

import pytest
import torch

import mfcl.utils.dist as mdist


def test_reduce_dict_mean_and_sum_monkeypatched():
    # Monkeypatch distributed environment
    fake_dist = types.SimpleNamespace()

    def all_reduce(tensor, op=None):
        # Simulate two processes sum: multiply by 2
        tensor.mul_(2)

    fake_dist.all_reduce = all_reduce

    orig_dist = mdist.dist
    orig_is_dist = mdist.is_dist
    orig_get_ws = mdist.get_world_size
    try:
        mdist.dist = fake_dist  # type: ignore
        mdist.is_dist = lambda: True  # type: ignore
        mdist.get_world_size = lambda: 2  # type: ignore
        d = {"a": torch.tensor(1.0)}
        out_mean = mdist.reduce_dict(d, op="mean")
        # Mean over 2 ranks should give original value
        assert torch.allclose(out_mean["a"], torch.tensor(1.0))
        out_sum = mdist.reduce_dict(d, op="sum")
        assert torch.allclose(out_sum["a"], torch.tensor(2.0))
    finally:
        mdist.dist = orig_dist
        mdist.is_dist = orig_is_dist  # type: ignore
        mdist.get_world_size = orig_get_ws  # type: ignore


def test_reduce_dict_invalid_op_raises():
    with pytest.raises(ValueError):
        mdist.reduce_dict({"a": torch.tensor(1.0)}, op="median")
