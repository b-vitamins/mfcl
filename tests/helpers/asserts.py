from __future__ import annotations

import torch


def assert_close(a: torch.Tensor, b: torch.Tensor, *, rtol=1e-4, atol=1e-6) -> None:
    if a.dtype != torch.float32:
        a = a.float()
    if b.dtype != torch.float32:
        b = b.float()
    diff = torch.max(torch.abs(a - b))
    if diff > atol + rtol * torch.max(torch.ones_like(b), torch.abs(b)).max():
        raise AssertionError(f"Tensors differ by max {diff}")


def assert_no_nan(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if torch.isnan(t).any():
            raise AssertionError("NaNs found in tensor")


def assert_unit_norm(x: torch.Tensor, tol: float = 1e-5) -> None:
    n = torch.norm(x, dim=1)
    if torch.max(torch.abs(n - 1.0)) > tol:
        raise AssertionError("Rows are not unit norm within tolerance")
