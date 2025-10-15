import torch

from mfcl.data.collate import collate_pair, collate_multicrop, collate_linear


def test_collate_pair_shapes():
    batch = [
        ({"view1": torch.randn(3, 8, 8), "view2": torch.randn(3, 8, 8)}, i)
        for i in range(4)
    ]
    out = collate_pair(batch)
    assert out["view1"].shape == (4, 3, 8, 8)
    assert out["view2"].shape == (4, 3, 8, 8)
    assert out["index"].dtype == torch.long


def test_collate_multicrop_shapes_and_mismatch():
    sample = {
        "crops": [torch.randn(3, 8, 8), torch.randn(3, 4, 4)],
        "code_crops": (0, 1),
    }
    batch_ok = [(sample, 0), (sample, 1)]
    out = collate_multicrop(batch_ok)
    assert len(out["crops"]) == 2
    assert out["crops"][0].shape == (2, 3, 8, 8)
    bad = {"crops": [torch.randn(3, 9, 9), torch.randn(3, 4, 4)], "code_crops": (0, 1)}
    try:
        collate_multicrop([(sample, 0), (bad, 1)])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_collate_linear():
    batch = [(torch.randn(3, 8, 8), 1), (torch.randn(3, 8, 8), 0)]
    out = collate_linear(batch)
    assert out["input"].shape == (2, 3, 8, 8)
    assert out["target"].dtype == torch.long
