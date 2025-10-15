"""Atomic checkpoint save/load with retention and latest symlink.

This module is trainer-agnostic. It provides simple helpers to save and load
training state dicts (model/optim/sched/scaler/epoch/etc.) with atomic writes,
keep-K pruning, and a convenient ``latest.pt`` pointer.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Dict, Optional

import torch

StateDict = Dict[str, Any]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _atomic_save(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup of temp file on failure
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def _derive_prune_pattern(path: str) -> str:
    """Derive a glob pattern for retention based on basename prefix.

    Example: '.../ckpt_ep0042.pt' -> pattern 'ckpt_ep*.pt'
    Fallback: '<name>*.ext' if no digits found.
    """
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    # Take leading non-digit prefix
    i = 0
    while i < len(name) and not name[i].isdigit():
        i += 1
    prefix = name[:i] if i > 0 else name
    if not prefix:
        prefix = name
    return f"{prefix}*{ext}"


def _prune_old_checkpoints(path: str, keep_k: int) -> None:
    if keep_k <= 0:
        return
    d = os.path.dirname(os.path.abspath(path)) or "."
    pattern = _derive_prune_pattern(path)
    all_matches = [
        p
        for p in glob.glob(os.path.join(d, pattern))
        if os.path.isfile(p) and os.path.basename(p) != "latest.pt"
    ]
    if not all_matches:
        return
    all_matches.sort()
    to_remove = all_matches[:-keep_k]
    for p in to_remove:
        try:
            os.remove(p)
        except Exception as e:
            raise RuntimeError(f"Failed to prune old checkpoint: {p}") from e


def _update_latest_symlink(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    latest = os.path.join(d, "latest.pt")
    try:
        if os.path.lexists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except (AttributeError, NotImplementedError, OSError):
        # Fallback to copy if symlinks not supported.
        shutil.copy2(path, latest)


def save_checkpoint(
    path: str,
    state: StateDict,
    keep_k: int = 3,
    make_latest: bool = True,
) -> str:
    """Atomically save a checkpoint and prune older ones.

    Args:
        path: Target file path, e.g. 'runs/exp1/ckpt_ep0042.pt'.
        state: Arbitrary dict (must be torch.save-able).
        keep_k: Number of most recent checkpoints to retain in the directory.
            If <= 0, do not delete any.
        make_latest: If True, create/update symlink 'latest.pt' in the same dir.

    Returns:
        The final saved path (identical to 'path').

    Raises:
        OSError: On filesystem errors (create, rename, symlink).
        RuntimeError: If save partially succeeds but pruning fails in a way that
            leaves temp files behind.
    """
    _ensure_dir(path)
    _atomic_save(path, state)
    try:
        _prune_old_checkpoints(path, keep_k)
    finally:
        # Ensure latest pointer is updated even if pruning raises; this matches
        # the expectation that the newest checkpoint is usable.
        if make_latest:
            _update_latest_symlink(path)
    return path


def load_checkpoint(
    path: str,
    map_location: Optional[str | torch.device] = None,
    strict: bool = False,
) -> StateDict:
    """Load a checkpoint dict via ``torch.load``.

    Args:
        path: File path to load.
        map_location: torch.load map_location.
        strict: If True, raise FileNotFoundError; if False, return {} when missing.

    Returns:
        The loaded state dict (possibly empty if strict=False and missing).

    Raises:
        FileNotFoundError: If strict=True and path does not exist.
        OSError: On filesystem errors.
        RuntimeError: If the object cannot be deserialized.
    """
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(path)
        return {}
    return torch.load(path, map_location=map_location)


def latest_checkpoint(dir_path: str, pattern: str = "ckpt_ep*.pt") -> Optional[str]:
    """Return the lexicographically latest checkpoint path in a directory.

    Args:
        dir_path: Directory to scan.
        pattern: Glob pattern (default matches 'ckpt_epXXXX.pt').

    Returns:
        Path to latest checkpoint or None if none found.
    """
    matches = [
        p for p in glob.glob(os.path.join(dir_path, pattern)) if os.path.isfile(p)
    ]
    if not matches:
        return None
    matches.sort()
    return matches[-1]


__all__ = ["save_checkpoint", "load_checkpoint", "latest_checkpoint", "StateDict"]
