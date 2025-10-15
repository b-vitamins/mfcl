from __future__ import annotations

import contextlib


@contextlib.contextmanager
def fake_dist_env():
    """A no-op context for potential future dist harness.

    Placeholder: In CI, you can replace with a true 2-proc gloo spawn.
    """
    yield
