"""Common image transform primitives with explicit parameters.

Provides deterministic normalization helpers, a simple Solarize transform,
Gaussian kernel sizing, and small wrappers convenient for building pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

from torchvision import transforms as T


if TYPE_CHECKING:
    from PIL import Image


def normalize_stats() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return ImageNet mean and std as tuples.

    Returns:
        (mean, std): Each a 3-tuple of floats in RGB order.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def gaussian_kernel_size(img_size: int, frac: float = 0.10) -> int:
    """Compute an odd Gaussian kernel size proportional to image size.

    Args:
        img_size: Target crop size in pixels.
        frac: Fraction of the width used for kernel size (typ. 0.10).

    Returns:
        Odd integer >= 3; e.g., ``img_size=160`` with ``frac=0.1`` produces
        a kernel of approximately 17.

    Raises:
        ValueError: If img_size < 16 or frac <= 0.
    """
    if img_size < 16:
        raise ValueError("img_size must be >= 16")
    if frac <= 0:
        raise ValueError("frac must be > 0")
    # Ensure odd and at least 3
    k = int(frac * img_size)
    k = k | 1  # make odd
    if k < 3:
        k = 3
    return k


class Solarize:
    """Deterministic solarize transform compatible with PIL images.

    Implements inversion of pixel values strictly greater than a threshold,
    leaving others unchanged, in a version-agnostic way that does not rely on
    the exact semantics of the underlying PIL ImageOps implementation.
    Pixels strictly greater than threshold are inverted; others unchanged.
    """

    def __init__(self, threshold: int = 128):
        if not (0 <= int(threshold) <= 255):
            raise ValueError("threshold must be in [0,255]")
        self.threshold = int(threshold)

    def __call__(self, img: "Image.Image") -> "Image.Image":
        try:
            from PIL import Image
        except Exception as e:  # pragma: no cover - PIL is a transitive dep
            raise ImportError("PIL is required for Solarize") from e
        # Convert to numpy for robust manipulation regardless of PIL version
        import numpy as np

        arr = np.array(img, dtype=np.uint8, copy=True)
        mask = arr > self.threshold
        arr[mask] = 255 - arr[mask]
        return Image.fromarray(arr, mode=getattr(img, "mode", "RGB"))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


def _find_solarize_in_compose(comp: T.Compose) -> list[Solarize]:
    """Return all Solarize transforms contained (recursively) in a Compose."""

    def _collect(transform: object) -> list[Solarize]:
        found: list[Solarize] = []
        if isinstance(transform, Solarize):
            found.append(transform)
        inner = getattr(transform, "transforms", None)
        if inner is not None:
            for sub in inner:
                found.extend(_collect(sub))
        return found

    result: list[Solarize] = []
    for t in comp.transforms:
        result.extend(_collect(t))
    return result


def build_color_jitter(strength: float) -> T.ColorJitter:
    """Return a ColorJitter with equal RGB and brightness/contrast levels.

    Args:
        strength: Base jitter strength (e.g., 0.4 for SimCLR).

    Returns:
        torchvision ColorJitter instance with (b=c=sat=strength, hue=0.1).
    """
    s = float(strength)
    return T.ColorJitter(brightness=s, contrast=s, saturation=s, hue=0.1)


def normalize_tensor() -> T.Normalize:
    """Return torchvision Normalize with ImageNet stats."""
    mean, std = normalize_stats()
    return T.Normalize(mean=mean, std=std)


def to_tensor_and_norm() -> T.Compose:
    """Return PILToTensor -> float32 -> Normalize composition.

    Uses PILToTensor + ConvertImageDtype to avoid occasional issues with
    ToTensor in certain environments.
    """
    import torch

    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        normalize_tensor(),
    ])


def random_apply(transform: Callable, p: float) -> T.RandomApply:
    """Wrap a transform in RandomApply with probability p.

    Args:
        transform: Transform callable.
        p: Application probability in [0,1].
    """
    return T.RandomApply([transform], p=float(p))


__all__ = [
    "normalize_stats",
    "gaussian_kernel_size",
    "Solarize",
    "build_color_jitter",
    "normalize_tensor",
    "to_tensor_and_norm",
    "random_apply",
]
