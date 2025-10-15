"""Fixed-size FIFO ring buffer for 2D tensors (MoCo-style queue)."""

from __future__ import annotations


import torch


class RingQueue:
    """Fixed-size FIFO ring buffer for 2D tensors.

    Stores features in a [size, dim] buffer on a chosen device. ``enqueue``
    appends rows cyclically. ``get`` returns the current content in storage
    order (no rotation to chronological order is performed).
    """

    def __init__(self, dim: int, size: int, device: str | torch.device = "cpu") -> None:
        """Construct a ring queue.

        Args:
            dim: Feature dimension D.
            size: Max entries K.
            device: Storage device (cpu recommended).
        """
        if dim <= 0 or size <= 0:
            raise ValueError("dim and size must be > 0")
        self.dim = int(dim)
        self.size = int(size)
        self.ptr = 0
        self.full = False
        self.buf = torch.zeros(self.size, self.dim, device=device, dtype=torch.float32)

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor) -> None:
        """Append rows of x (N,D), overwriting oldest entries cyclically."""
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x shape [N,{self.dim}], got {tuple(x.shape)}")
        x = x.to(device=self.buf.device, dtype=self.buf.dtype, copy=False)
        n = x.shape[0]
        if n == 0:
            return
        end = self.ptr + n
        if end <= self.size:
            self.buf[self.ptr : end].copy_(x)
        else:
            first = self.size - self.ptr
            self.buf[self.ptr :].copy_(x[:first])
            self.buf[: end % self.size].copy_(x[first:])
        self.ptr = (self.ptr + n) % self.size
        if n >= self.size or self.ptr == 0:
            self.full = True

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        """Return current buffer content [K', D] with K' = size if full else ptr."""
        if self.full:
            return self.buf
        return self.buf[: self.ptr]


__all__ = ["RingQueue"]
