from __future__ import annotations

import os
from typing import Tuple, Optional, Union

import numpy as np
import torch


def _to_device_dtype(
    t: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        ten = torch.from_numpy(t)
    else:
        ten = t
    ten = ten.to(dtype=dtype)
    if device is not None:
        ten = ten.to(device, non_blocking=True)
    return ten


def load_tensor(
    path: str,
    device: Optional[torch.device] = None,
    mmap: bool = True,
    key: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a single tensor from .pt/.pth or .npz/.npy.
    - For .pt/.pth: expects a tensor or a dict containing a tensor.
    - For .npz: uses key if provided, else first array.
    - For .npy: loads array directly.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if key is None:
                for k, v in obj.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        obj = v
                        break
                else:
                    raise KeyError("No tensor-like entry found in checkpoint")
            else:
                obj = obj[key]
        if not isinstance(obj, (torch.Tensor, np.ndarray)):
            raise TypeError("Expected tensor-like object in checkpoint")
        return _to_device_dtype(obj, device, dtype)
    elif ext == ".npz":
        with np.load(path, mmap_mode="r" if mmap else None) as z:
            if key is None:
                first_key = list(z.keys())[0]
                arr = z[first_key]
            else:
                arr = z[key]
        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected numpy array in npz file")
        return _to_device_dtype(arr, device, dtype)
    elif ext == ".npy":
        arr = np.load(path, mmap_mode="r" if mmap else None)
        return _to_device_dtype(arr, device, dtype)
    else:
        raise ValueError(f"Unsupported extension for tensor: {path}")


def load_pair(
    x_path: str,
    y_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    mmap: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a pair of embedding matrices X, Y with matching row counts.
    Supports:
      - separate files for X and Y
      - a single file containing dict with 'X' and 'Y'
    """
    if y_path is None:
        # Single file case; expect dict with 'X' and 'Y'
        ext = os.path.splitext(x_path)[1].lower()
        if ext in [".pt", ".pth"]:
            obj = torch.load(x_path, map_location="cpu")
            X = obj["X"] if "X" in obj else obj.get("x") or obj.get("emb_x")
            Y = obj["Y"] if "Y" in obj else obj.get("y") or obj.get("emb_y")
            if X is None or Y is None:
                raise KeyError("Single file must contain 'X' and 'Y'.")
        elif ext == ".npz":
            with np.load(x_path, mmap_mode="r" if mmap else None) as z:
                keys = set(z.keys())
                xk = "X" if "X" in keys else ("x" if "x" in keys else None)
                yk = "Y" if "Y" in keys else ("y" if "y" in keys else None)
                if xk is None or yk is None:
                    raise KeyError("Single NPZ must contain arrays 'X' and 'Y'.")
                X = z[xk]
                Y = z[yk]
        else:
            raise ValueError("Single-file pair must be .pt/.pth or .npz")
    else:
        X = load_tensor(x_path, device=None, mmap=mmap, dtype=torch.float32)
        Y = load_tensor(y_path, device=None, mmap=mmap, dtype=torch.float32)

    if not isinstance(X, (torch.Tensor, np.ndarray)) or not isinstance(
        Y, (torch.Tensor, np.ndarray)
    ):
        raise TypeError("Expected tensor-like objects for X and Y")

    # Ensure torch tensors, correct dtype/device
    X = _to_device_dtype(X, device, dtype)
    Y = _to_device_dtype(Y, device, dtype)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D matrices [N, d].")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Row mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]} rows."
        )

    return X, Y


def pick_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
