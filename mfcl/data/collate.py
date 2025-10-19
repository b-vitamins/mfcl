"""Collate functions for SSL two-view and multi-crop batches and linear eval."""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Sequence, cast

import torch


def collate_pair(
    batch: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """Collate a batch of two-view dicts or GPU-ready single images."""

    if not batch:
        raise ValueError("batch must contain at least one element")
    sample0, _ = batch[0]
    idxs = [int(idx) for _, idx in batch]
    if "view1" in sample0 and "view2" in sample0:
        v1 = [sample["view1"] for sample, _ in batch]
        v2 = [sample["view2"] for sample, _ in batch]
        return {
            "view1": torch.stack(v1, dim=0),
            "view2": torch.stack(v2, dim=0),
            "index": torch.tensor(idxs, dtype=torch.long),
        }
    if "image" in sample0:
        images = [sample["image"] for sample, _ in batch]
        return {
            "image": images,
            "index": torch.tensor(idxs, dtype=torch.long),
        }
    raise KeyError("Expected keys 'view1'/'view2' or 'image' in samples")

class MultiCropSample(TypedDict):
    crops: List[torch.Tensor]
    code_crops: Tuple[int, int]


class MultiCropBatch(TypedDict):
    crops: List[torch.Tensor]
    code_crops: Tuple[int, int]
    index: torch.Tensor


def collate_multicrop(batch: Sequence[Tuple[Dict[str, object], int]]) -> MultiCropBatch:
    """Collate a batch of multi-crop samples or GPU-ready base images."""

    if not batch:
        raise ValueError("batch must contain at least one element")

    first, _ = batch[0]
    if "image" in first:
        images = [sample["image"] for sample, _ in batch]
        idxs = torch.tensor([int(idx) for _, idx in batch], dtype=torch.long)
        return {"image": torch.stack(images, dim=0), "index": idxs}

    f = cast(MultiCropSample, first)
    crops0 = f.get("crops")
    if not isinstance(crops0, list):
        raise TypeError("sample['crops'] must be a list of tensors")
    n_crops = len(crops0)
    # Validate code_crops consistency
    code_crops = f.get("code_crops")
    if not isinstance(code_crops, tuple) or len(code_crops) != 2:
        raise ValueError("sample['code_crops'] must be a tuple of length 2")
    a, b = code_crops
    if not (0 <= a < n_crops and 0 <= b < n_crops and a != b):
        raise ValueError("code_crops must be distinct indices within [0, n_crops)")
    for sample, _ in batch[1:]:
        s = cast(MultiCropSample, sample)
        if s.get("code_crops") != code_crops:
            raise ValueError("code_crops mismatch across batch")

    stacked: List[torch.Tensor] = []
    for i in range(n_crops):
        tensors_i = []
        size_ref = None
        for sample, _ in batch:
            s = cast(MultiCropSample, sample)
            crops = s.get("crops")
            if not isinstance(crops, list):
                raise TypeError("sample['crops'] must be a list of tensors")
            if i >= len(crops):
                raise ValueError("inconsistent number of crops across batch")
            t = crops[i]
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"crop at position {i} must be a Tensor")
            if size_ref is None:
                size_ref = tuple(t.shape)
            elif tuple(t.shape) != size_ref:
                raise ValueError(f"crop size mismatch at position {i}")
            tensors_i.append(t)
        stacked.append(torch.stack(tensors_i, dim=0))

    idxs = torch.tensor([int(idx) for _, idx in batch], dtype=torch.long)
    return {"crops": stacked, "code_crops": code_crops, "index": idxs}


def collate_linear(batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    """Collate a batch for linear evaluation.

    Args:
        batch: List of (input_tensor, label) pairs.

    Returns:
        {'input': Tensor[B,3,H,W], 'target': LongTensor[B]}
    """
    inputs = [x for x, _ in batch]
    targets = [int(y) for _, y in batch]
    return {
        "input": torch.stack(inputs, dim=0),
        "target": torch.tensor(targets, dtype=torch.long),
    }


__all__ = ["collate_pair", "collate_multicrop", "collate_linear"]
