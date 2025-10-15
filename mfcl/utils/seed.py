"""Seeding utilities for reproducible runs.

This module provides a single entry point to seed Python's ``random``,
NumPy's RNG, and PyTorch (CPU and CUDA), and to configure cuDNN
determinism/benchmark flags consistently.
"""

from __future__ import annotations

import random


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed all RNGs and configure cuDNN determinism.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, enforce deterministic cuDNN kernels.
        benchmark: If True, enable cuDNN benchmark (ignored if deterministic).

    Raises:
        ValueError: If ``seed`` is negative.
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
