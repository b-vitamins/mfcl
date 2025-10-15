"""Seeding utilities for reproducible runs.

This module provides a single entry point to seed Python's ``random``,
NumPy's RNG, and PyTorch (CPU and CUDA), and to configure cuDNN
determinism/benchmark flags consistently.
"""

from __future__ import annotations

import random


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    strict: bool = False,
    allow_tf32: bool | None = None,
) -> None:
    """Seed all RNGs and configure determinism and precision trade-offs.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, prefer deterministic cuDNN kernels.
        benchmark: If True, enable cuDNN benchmark (ignored if deterministic).
        strict: If True, request fully deterministic torch algorithms where available.
        allow_tf32: Optional toggle for TF32 matmul/cuDNN usage; None leaves as-is.

    Raises:
        ValueError: If ``seed`` is negative.

    Notes:
        ``deterministic=True`` prefers deterministic cuDNN kernels but may reduce
        performance. Setting ``strict=True`` additionally requests deterministic
        algorithm implementations globally, potentially raising errors when not
        available. ``allow_tf32`` toggles TF32 usage, affecting numerical
        precision/performance without guaranteeing determinism.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        # NumPy is optional; ignore if unavailable.
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if strict and hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        if allow_tf32 is not None:
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            except Exception:
                pass

        # cuDNN determinism/benchmark configuration
        try:
            import torch.backends.cudnn as cudnn

            if deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.deterministic = False
                cudnn.benchmark = bool(benchmark)
        except Exception:
            # Backends may not be present in some builds.
            pass
    except Exception:
        # Torch may not be installed at doc-build time.
        pass


__all__ = ["set_seed"]
