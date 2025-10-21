"""Fixed-size FIFO ring buffer for 2D tensors (MoCo-style queue)."""

from __future__ import annotations


import torch


class RingQueue:
    """Fixed-size FIFO ring buffer for 2D tensors.

    Args:
        dim: Feature dimension D.
        size: Maximum number of rows the queue stores.
        device: Device that backs the underlying tensor buffer.
        dtype: Tensor dtype used for storage.

    Returns:
        ``None``.

    Raises:
        ValueError: If ``dim`` or ``size`` are not positive.
    """

    def __init__(
        self,
        dim: int,
        size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Construct a ring queue.

        Args:
            dim: Feature dimension D.
            size: Max entries K.
            device: Storage device (cpu recommended).
            dtype: Torch dtype used to allocate the buffer.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``dim`` or ``size`` are not positive.
        """
        if dim <= 0 or size <= 0:
            raise ValueError("dim and size must be > 0")
        self.dim = int(dim)
        self.size = int(size)
        self.ptr = 0
        self.full = False
        self.buf = torch.zeros(self.size, self.dim, device=device, dtype=dtype)

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor) -> None:
        """Append rows of ``x`` overwriting the oldest entries cyclically.

        Args:
            x: Tensor with shape ``[N, dim]`` to append.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``x`` does not have shape ``[N, dim]``.
        """
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
        """Return the currently stored rows in storage order.

        Returns:
            Tensor of shape ``[K', dim]`` where ``K'`` is ``size`` if the
            buffer has been filled, otherwise ``ptr``.
        """
        if self.full:
            return self.buf
        return self.buf[: self.ptr]

    def __len__(self) -> int:
        """Return the number of valid entries currently stored.

        Returns:
            Integer count of valid entries.
        """

        return self.size if self.full else self.ptr

    @torch.no_grad()
    def clear(self, zero: bool = False) -> None:
        """Reset queue to empty; optionally zero the buffer.

        Args:
            zero: Whether to fill the buffer with zeros after clearing.

        Returns:
            ``None``.
        """

        self.ptr = 0
        self.full = False
        if zero:
            self.buf.zero_()


MoCoQueue = RingQueue


__all__ = ["RingQueue", "MoCoQueue"]
