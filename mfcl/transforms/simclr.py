"""Two-view transform pipeline for SimCLR/BYOL/SimSiam family."""

from __future__ import annotations

from typing import Callable, Dict

from torchvision import transforms as T

from mfcl.core.config import AugConfig
from mfcl.transforms.common import (
    build_color_jitter,
    gaussian_kernel_size,
    Solarize,
    to_tensor_and_norm,
    random_apply,
)


def build_pair_transforms(cfg: AugConfig) -> Callable:
    """Return a callable that maps a PIL image -> {'view1': Tensor, 'view2': Tensor}.

    Args:
        cfg: AugConfig with fields:
            - img_size: int
            - jitter_strength: float
            - blur_prob: float
            - gray_prob: float
            - solarize_prob: float
            - solarize_threshold: Optional[int]; defaults to 128 when absent.

    Returns:
        A function f(img) -> {'view1': Tensor[C,H,W], 'view2': Tensor[C,H,W]} where
        H = W = cfg.img_size, normalized to ImageNet stats.
    """
    k = gaussian_kernel_size(cfg.img_size)
    color = build_color_jitter(cfg.jitter_strength)

    chain = [
        T.RandomResizedCrop(cfg.img_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.RandomApply([color], p=0.8),
        T.RandomGrayscale(p=float(cfg.gray_prob)),
        random_apply(
            T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0)), p=float(cfg.blur_prob)
        ),
    ]
    if getattr(cfg, "solarize_prob", 0.0) and cfg.solarize_prob > 0:
        thresh = int(getattr(cfg, "solarize_threshold", 128))
        chain.append(random_apply(Solarize(thresh), p=float(cfg.solarize_prob)))
    chain.append(to_tensor_and_norm())
    transform = T.Compose(chain)

    def two_views(img) -> Dict[str, object]:
        return {"view1": transform(img), "view2": transform(img)}

    return two_views


__all__ = ["build_pair_transforms"]
