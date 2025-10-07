from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


@dataclass
class ByteCounters:
    all_reduce_payload_bytes: int = 0
    all_reduce_theoretical_bytes: int = 0
    all_gather_payload_bytes: int = 0
    all_gather_theoretical_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "bytes_all_reduce_payload": int(self.all_reduce_payload_bytes),
            "bytes_all_reduce_theoretical": int(self.all_reduce_theoretical_bytes),
            "bytes_all_gather_payload": int(self.all_gather_payload_bytes),
            "bytes_all_gather_theoretical": int(self.all_gather_theoretical_bytes),
        }


class DistEnv:
    """
    Minimal distributed environment helper with byte accounting.
    Initializes from env:// if WORLD_SIZE>1.
    """

    def __init__(self) -> None:
        self.initialized = False
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bytes = ByteCounters()

    def init(self, backend: Optional[str] = None) -> None:
        if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if dist is None:
                raise RuntimeError("torch.distributed is not available")
            if not dist.is_initialized():
                if backend is None:
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
            self.initialized = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                self.device = torch.device("cuda", local_rank)
        else:
            self.initialized = False
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def barrier(self) -> None:
        if self.initialized and dist is not None and dist.is_initialized():
            dist.barrier()

    def cleanup(self) -> None:
        if self.initialized and dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        self.initialized = False
        self.rank = 0
        self.world_size = 1

    def _theoretical_bytes_ring(self, num_bytes: int) -> int:
        p = max(self.world_size, 1)
        return int(2 * (p - 1) / p * num_bytes)

    def all_reduce(
        self, tensor: torch.Tensor, op: Optional[str] = "sum"
    ) -> torch.Tensor:
        if not self.initialized:
            return tensor
        if dist is None or not dist.is_initialized():
            return tensor
        num_bytes = tensor.numel() * tensor.element_size()
        self.bytes.all_reduce_payload_bytes += num_bytes
        self.bytes.all_reduce_theoretical_bytes += self._theoretical_bytes_ring(
            num_bytes
        )

        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "avg":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= float(self.world_size)
        else:
            dist.all_reduce(tensor)
        return tensor

    def all_gather_concat(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            return tensor
        if dist is None or not dist.is_initialized():
            return tensor
        num_bytes = tensor.numel() * tensor.element_size()
        self.bytes.all_gather_payload_bytes += num_bytes
        self.bytes.all_gather_theoretical_bytes += self._theoretical_bytes_ring(
            num_bytes
        )

        gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, tensor)
        return torch.cat(gather_list, dim=0)

    def reset_bytes(self) -> None:
        self.bytes = ByteCounters()

    def get_bytes(self) -> ByteCounters:
        return self.bytes
