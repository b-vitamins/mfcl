"""mfcl: Minimal, modular self-supervised learning components.

This package exposes registries and factories for building encoders, heads,
losses, methods, and data pipelines. Import order is deterministic and no
module performs side-effectful registration at import time.
"""

from . import core  # re-export subpackage for convenience

__all__ = [
    "core",
]
