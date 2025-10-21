"""Wrappers around ``torch.distributed`` collectives with telemetry hooks."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.distributed as _dist

from mfcl.telemetry.comms_logger import (
    AsyncCollectiveHandle,
    PayloadCategory,
    get_comms_logger,
    is_logging_enabled,
    log_collective,
)

def _is_async_work(work: object) -> bool:
    """Return ``True`` when *work* implements the async collective protocol."""

    return callable(getattr(work, "wait", None)) and callable(
        getattr(work, "is_completed", None)
    )


def _wrap_async_collective(
    *,
    kind: str,
    tensor: torch.Tensor,
    category: PayloadCategory,
    work: Any,
    start_time_s: float,
):
    if not is_logging_enabled():
        return work
    logger = get_comms_logger()
    if logger is None:
        return work
    return AsyncCollectiveHandle(
        kind=kind,
        tensor=tensor,
        category=category,
        work=work,
        logger=logger,
        start_time_s=start_time_s,
    )


def _execute_collective(
    *,
    kind: str,
    tensor: torch.Tensor,
    category: PayloadCategory,
    launcher: Callable[[bool], object],
    async_op: bool,
) -> AsyncCollectiveHandle | object:
    if async_op:
        start_time = time.perf_counter()
        work = launcher(True)
        if _is_async_work(work):
            return _wrap_async_collective(
                kind=kind,
                tensor=tensor,
                category=category,
                work=work,
                start_time_s=start_time,
            )
        return work
    with log_collective(kind, tensor, category):
        return launcher(False)


def all_reduce(
    tensor: torch.Tensor,
    op: _dist.ReduceOp = _dist.ReduceOp.SUM,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    *,
    category: PayloadCategory = PayloadCategory.OTHER,
) -> AsyncCollectiveHandle | object:
    """All-reduce wrapper that records communication metadata.

    Args:
        tensor: Tensor that will participate in the all-reduce.
        op: Reduction operator passed to :func:`torch.distributed.all_reduce`.
        group: Process group that scopes the collective.
        async_op: When ``True``, return an asynchronous work handle.
        category: Semantic label describing the payload being reduced.

    Returns:
        ``None`` for synchronous execution, or an asynchronous work handle when
        ``async_op`` is ``True``. When communication logging is enabled the
        handle is an :class:`~mfcl.telemetry.comms_logger.AsyncCollectiveHandle`
        that finalises telemetry upon completion.
    """

    return _execute_collective(
        kind="all_reduce",
        tensor=tensor,
        category=category,
        launcher=lambda flag: _dist.all_reduce(tensor, op=op, group=group, async_op=flag),
        async_op=async_op,
    )


def all_gather(
    tensor_list: Sequence[torch.Tensor],
    tensor: torch.Tensor,
    *,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
) -> AsyncCollectiveHandle | object:
    """All-gather wrapper that records communication metadata.

    Args:
        tensor_list: Output buffers receiving data from each rank.
        tensor: Local tensor to contribute to the gather.
        group: Process group that scopes the collective.
        async_op: When ``True``, return an asynchronous work handle.
        category: Semantic label describing the payload being gathered.

    Returns:
        ``None`` for synchronous execution, or an asynchronous work handle when
        ``async_op`` is ``True``.
    """

    return _execute_collective(
        kind="all_gather",
        tensor=tensor,
        category=category,
        launcher=lambda flag: _dist.all_gather(
            tensor_list, tensor, group=group, async_op=flag
        ),
        async_op=async_op,
    )


def reduce_scatter(
    output: torch.Tensor,
    input_list: Iterable[torch.Tensor],
    *,
    op: _dist.ReduceOp = _dist.ReduceOp.SUM,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
) -> AsyncCollectiveHandle | object:
    """Reduce-scatter wrapper that records communication metadata.

    Args:
        output: Tensor that receives the reduced shard.
        input_list: Iterable of tensors to be reduced and scattered.
        op: Reduction operator passed to :func:`torch.distributed.reduce_scatter`.
        group: Process group that scopes the collective.
        async_op: When ``True``, return an asynchronous work handle.
        category: Semantic label describing the payload being reduced.

    Returns:
        ``None`` for synchronous execution, or an asynchronous work handle when
        ``async_op`` is ``True``.
    """

    return _execute_collective(
        kind="reduce_scatter",
        tensor=output,
        category=category,
        launcher=lambda flag: _dist.reduce_scatter(
            output, input_list, op=op, group=group, async_op=flag
        ),
        async_op=async_op,
    )


def broadcast(
    tensor: torch.Tensor,
    src: int,
    *,
    group: _dist.ProcessGroup | None = None,
    async_op: bool = False,
    category: PayloadCategory = PayloadCategory.OTHER,
) -> AsyncCollectiveHandle | object:
    """Broadcast wrapper that records communication metadata.

    Args:
        tensor: Tensor to broadcast.
        src: Rank of the source process.
        group: Process group that scopes the collective.
        async_op: When ``True``, return an asynchronous work handle.
        category: Semantic label describing the payload being broadcast.

    Returns:
        ``None`` for synchronous execution, or an asynchronous work handle when
        ``async_op`` is ``True``.
    """

    return _execute_collective(
        kind="broadcast",
        tensor=tensor,
        category=category,
        launcher=lambda flag: _dist.broadcast(tensor, src=src, group=group, async_op=flag),
        async_op=async_op,
    )


__all__ = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "PayloadCategory",
]
