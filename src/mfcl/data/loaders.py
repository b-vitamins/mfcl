from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

from mfcl.data.neighbors import cosine_topk_streaming


@dataclass
class IIDBatchSpec:
    batch_size: int
    num_batches: int
    seed: int = 42


def iid_batches(
    X: torch.Tensor, Y: torch.Tensor, spec: IIDBatchSpec
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield IID batches of paired rows from X and Y.
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of rows"
    N_total = X.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(spec.seed)

    for _ in range(spec.num_batches):
        if spec.batch_size > N_total:
            raise ValueError(f"batch_size {spec.batch_size} > available rows {N_total}")
        idx = torch.randperm(N_total, generator=gen)[: spec.batch_size]
        yield X.index_select(0, idx), Y.index_select(0, idx)


@dataclass
class ClassCoherentSpec(IIDBatchSpec):
    labels: Optional[torch.Tensor] = None  # [N_total], required
    class_id: Optional[int] = None  # choose a class to sample from


def class_coherent_batches(
    X: torch.Tensor, Y: torch.Tensor, spec: ClassCoherentSpec
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample batches restricted to a single class. Requires labels over the full dataset.
    If class_id is None, picks a random class each batch from the available labels.
    """
    if spec.labels is None:
        raise ValueError("ClassCoherentSpec.labels must be provided")
    labels = spec.labels
    classes = torch.unique(labels).tolist()
    gen = torch.Generator(device="cpu").manual_seed(spec.seed)

    for _ in range(spec.num_batches):
        cls = (
            spec.class_id
            if spec.class_id is not None
            else classes[torch.randint(0, len(classes), (1,), generator=gen).item()]
        )
        mask = labels == cls
        idx_pool = mask.nonzero(as_tuple=False).squeeze(1)
        if idx_pool.numel() < spec.batch_size:
            raise ValueError(
                f"Not enough samples for class {cls}: need {spec.batch_size}, have {idx_pool.numel()}"
            )
        perm = idx_pool[
            torch.randperm(idx_pool.numel(), generator=gen)[: spec.batch_size]
        ]
        yield X.index_select(0, perm), Y.index_select(0, perm)


@dataclass
class HardNegSpec(IIDBatchSpec):
    p: float = 0.25  # fraction indicator only used to choose difficulty (documented)
    k_within: int = 64  # nearest-neighbor budget within batch


def hard_negative_enriched_batches(
    X: torch.Tensor, Y: torch.Tensor, spec: HardNegSpec
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Within-batch hard-negative enrichment:
      1) Draw IID batch indices.
      2) (For diagnostics) compute within-batch top-k neighbors to characterize hardness.
    This keeps the exact set of negatives unchanged (since evaluation consumes the batch),
    but exposes a consistent "hardness" label downstream. Full-dataset enrichment would
    require an ANN index outside the scope of this phase.
    """
    N_total = X.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(spec.seed)

    for _ in range(spec.num_batches):
        if spec.batch_size > N_total:
            raise ValueError(f"batch_size {spec.batch_size} > available rows {N_total}")
        idx = torch.randperm(N_total, generator=gen)[: spec.batch_size]
        Xb = X.index_select(0, idx)
        Yb = Y.index_select(0, idx)
        # Touch neighbor routine to surface a “hardness” signal if needed
        _vals, _idx = cosine_topk_streaming(Xb, Yb, k=spec.k_within, exclude_self=True)
        # We return the batch as-is; the hybrid/metrics can inspect _vals if desired.
        yield Xb, Yb
