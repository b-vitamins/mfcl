"""SwAV-style multi-crop transform builder."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from torchvision import transforms as T

from mfcl.core.config import AugConfig
from mfcl.transforms.common import (
    build_color_jitter,
    gaussian_kernel_size,
    Solarize,
    to_tensor_and_norm,
    random_apply,
)


def _make_chain(size: int, scale: Tuple[float, float], cfg: AugConfig) -> T.Compose:
    k = gaussian_kernel_size(size)
    color = build_color_jitter(cfg.jitter_strength)
    chain = [
        T.RandomResizedCrop(size, scale=scale, ratio=(3 / 4, 4 / 3)),
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
    return T.Compose(chain)


def build_multicrop_transforms(
    cfg: AugConfig,
    global_scale: Tuple[float, float] = (0.14, 1.0),
    local_scale: Tuple[float, float] = (0.05, 0.14),
) -> Callable:
    """Return a callable that maps PIL image -> {'crops': List[Tensor], 'code_crops': Tuple[int,int]}.

    Args:
        cfg: AugConfig; uses img_size for global crops and local_size for local crops.
        global_scale: Scale range for global RandomResizedCrop.
        local_scale: Scale range for local RandomResizedCrop.

    Returns:
        f(img) -> {
          'crops': [Tensor[C,Hc,Wc], ...],   # global crops first, locals afterwards
          'code_crops': (0, 1)               # fixed indices of the two global crops
        }
    """
    if int(cfg.global_crops) < 2:
        raise ValueError("global_crops must be >= 2 for code_crops=(0,1)")

    g_chain = _make_chain(cfg.img_size, global_scale, cfg)
    l_chain = None
    if int(cfg.local_crops) > 0:
        l_chain = _make_chain(cfg.local_size, local_scale, cfg)

    def multi(img) -> Dict[str, object]:
        crops: List[object] = []
        for _ in range(int(cfg.global_crops)):
            crops.append(g_chain(img))
        if l_chain is not None:
            for _ in range(int(cfg.local_crops)):
                crops.append(l_chain(img))
        return {"crops": crops, "code_crops": (0, 1)}

    return multi


__all__ = ["build_multicrop_transforms"]
