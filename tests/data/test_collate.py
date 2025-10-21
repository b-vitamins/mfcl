import torch

from mfcl.data.collate import collate_pair


def test_collate_pair_two_view_mode():
    batch = [
        ({"view1": torch.randn(3, 6, 6), "view2": torch.randn(3, 6, 6)}, idx)
        for idx in range(3)
    ]

    result = collate_pair(batch)

    assert set(result.keys()) == {"view1", "view2", "index"}
    assert result["view1"].shape == (3, 3, 6, 6)
    assert result["view2"].shape == (3, 3, 6, 6)
    assert torch.equal(result["index"], torch.tensor([0, 1, 2], dtype=torch.long))


def test_collate_pair_single_image_mode_stacks():
    batch = [({"image": torch.randn(3, 5, 5)}, 4), ({"image": torch.randn(3, 5, 5)}, 5)]

    result = collate_pair(batch)

    assert set(result.keys()) == {"image", "index"}
    assert result["image"].shape == (2, 3, 5, 5)
    assert result["index"].dtype == torch.long
