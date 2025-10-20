"""Wrappers around ``torch.distributed`` collectives with telemetry hooks."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.distributed as _dist

from mfcl.telemetry.comms_logger import PayloadCategory, log_collective


def all_reduce(
    tensor: torch.Tensor,
    op: _dist.ReduceOp = _dist.ReduceOp.SUM,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    *,
    category: PayloadCategory = PayloadCategory.OTHER,
):
    """All-reduce wrapper that records communication metadata."""

    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported by MFCL comms wrappers yet."
        )
    with log_collective("all_reduce", tensor, category):
        return _dist.all_reduce(tensor, op=op, group=group, async_op=False)


def all_gather(
    tensor_list: Sequence[torch.Tensor],
    tensor: torch.Tensor,
    *,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
):
    """All-gather wrapper that records communication metadata."""

    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported by MFCL comms wrappers yet."
        )
    with log_collective("all_gather", tensor, category):
        return _dist.all_gather(tensor_list, tensor, group=group, async_op=False)


def reduce_scatter(
    output: torch.Tensor,
    input_list: Iterable[torch.Tensor],
    *,
    op: _dist.ReduceOp = _dist.ReduceOp.SUM,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
):
    """Reduce-scatter wrapper that records communication metadata."""

    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported by MFCL comms wrappers yet."
        )
    with log_collective("reduce_scatter", output, category):
        return _dist.reduce_scatter(
            output, input_list, op=op, group=group, async_op=False
        )


def broadcast(
    tensor: torch.Tensor,
    src: int,
    *,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
):
    """Broadcast wrapper that records communication metadata."""

    if async_op:
        raise NotImplementedError(
            "async_op=True is not supported by MFCL comms wrappers yet."
        )
    with log_collective("broadcast", tensor, category):
        return _dist.broadcast(tensor, src=src, group=group, async_op=False)


__all__ = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "PayloadCategory"]
