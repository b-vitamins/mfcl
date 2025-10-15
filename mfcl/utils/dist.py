"""Minimal distributed helpers compatible with torchrun.

Provides safe detection of distributed context, initialization from environment
variables (RANK, WORLD_SIZE, LOCAL_RANK), and simple collectives that degrade
to no-ops in single-process mode.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl", timeout_seconds: int = 1800) -> bool:
    """Initialize torch.distributed if environment variables are present.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment as
    set by ``torchrun``.

    Args:
        backend: 'nccl' (GPU) or 'gloo' (CPU only).
        timeout_seconds: Process group timeout.

    Returns:
        True if distributed was initialized, False if single-process.

    Raises:
        RuntimeError: On failed initialization when env variables suggest DDP.
    """
    world = int(os.getenv("WORLD_SIZE", "1"))
    if world <= 1:
        return False

    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Select device if CUDA is available; fall back to gloo on CPU.
    chosen_backend = backend
    if not torch.cuda.is_available() and backend == "nccl":
        chosen_backend = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(
            backend=chosen_backend,
            timeout=timedelta(seconds=int(timeout_seconds)),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize distributed process group (backend={chosen_backend})."
        ) from e
    return True


def is_dist() -> bool:
    """Return True if dist.is_available() and dist.is_initialized()."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return process rank (0 if not initialized)."""
    if is_dist():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Return LOCAL_RANK from env (0 if absent)."""

    try:
        return int(os.getenv("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def get_world_size() -> int:
    """Return world size (1 if not initialized)."""
    if is_dist():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True if rank == 0 (or not initialized)."""
    return get_rank() == 0


def barrier() -> None:
    """dist.barrier() when initialized, otherwise no-op."""
    if is_dist():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from src to all ranks.

    Args:
        obj: Python object to broadcast (ignored on non-src ranks).
        src: Source rank.

    Returns:
        The broadcasted object on all ranks.
    """
    if not is_dist():
        return obj
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def reduce_dict(
    tensors: Dict[str, torch.Tensor], op: str = "mean"
) -> Dict[str, torch.Tensor]:
    """All-reduce a dict of tensors across processes.

    Args:
        tensors: Mapping from name to tensor (all same shape/dtype).
        op: 'mean' or 'sum'.

    Returns:
        Reduced dict on every rank.
    """
    op_normalized = op.lower()
    if op_normalized not in {"mean", "sum"}:
        raise ValueError("op must be 'mean' or 'sum'")

    if not is_dist():
        return {k: v.detach().clone() for k, v in tensors.items()}

    reduced: Dict[str, torch.Tensor] = {}
    world = get_world_size()
    for k, v in tensors.items():
        if not torch.is_tensor(v):  # pragma: no cover - defensive
            raise TypeError(f"Value for key '{k}' must be a torch.Tensor")
        t = v.detach().clone()
        reduce_op = getattr(dist, "ReduceOp", None)
        if reduce_op is not None and hasattr(reduce_op, "SUM"):
            dist.all_reduce(t, op=reduce_op.SUM)
        else:  # pragma: no cover - defensive for mocked collectives
            dist.all_reduce(t)
        if op_normalized == "mean":
            t /= world
        reduced[k] = t
    return reduced


def all_gather_tensor(tensor: torch.Tensor, debug_shapes: bool = False) -> torch.Tensor:
    """All-gather a tensor along the first dimension.

    Args:
        tensor: Local tensor to gather (same shape across ranks).
        debug_shapes: When True, assert equal shapes across ranks before gather.

    Returns:
        Concatenated tensor across ranks (on current device).
    """
    if not is_dist():
        return tensor.clone()

    world = get_world_size()

    if debug_shapes:
        shape_tensor = torch.tensor(
            list(tensor.shape), device=tensor.device, dtype=torch.long
        )
        gathered_shapes = [torch.zeros_like(shape_tensor) for _ in range(world)]
        dist.all_gather(gathered_shapes, shape_tensor)
        reference = gathered_shapes[0]
        for idx, shape in enumerate(gathered_shapes[1:], start=1):
            if not torch.equal(shape, reference):
                raise RuntimeError(
                    f"all_gather_tensor: tensor shape mismatch across ranks (rank 0 {tuple(reference.tolist())} vs rank {idx} {tuple(shape.tolist())})"
                )

    tensors = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


def cleanup() -> None:
    """Destroy process group if initialized."""
    if is_dist():
        dist.destroy_process_group()


__all__ = [
    "init_distributed",
    "is_dist",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "broadcast_object",
    "reduce_dict",
    "all_gather_tensor",
    "get_local_rank",
    "cleanup",
]
