"""Tensor-based augmentation pipelines for GPU execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

try:  # torchvision is optional for unit tests without GPU
    from torchvision.transforms import v2
except ImportError as exc:  # pragma: no cover - handled at call site
    raise RuntimeError(
        "torchvision with transforms.v2 is required for mfcl.transforms.gpu"
    ) from exc

from mfcl.core.config import AugConfig
from mfcl.transforms.common import gaussian_kernel_size, normalize_stats


def _to_tensor_uint8() -> Callable[[Any], torch.Tensor]:
    """Return a callable decoding PIL images to uint8 tensors."""

    from torchvision import transforms as T

    return T.PILToTensor()


def build_gpu_pair_pretransform(cfg: AugConfig) -> Callable[[Any], Dict[str, torch.Tensor]]:
    """Return a minimal CPU transform that only decodes to tensor."""

    to_tensor = _to_tensor_uint8()

    def _transform(img: Any) -> Dict[str, torch.Tensor]:
        tensor = to_tensor(img)
        return {"image": tensor}

    return _transform


def build_gpu_multicrop_pretransform(cfg: AugConfig) -> Callable[[Any], Dict[str, torch.Tensor]]:
    """Return a minimal CPU transform for multi-crop GPU augmentation."""

    to_tensor = _to_tensor_uint8()

    def _transform(img: Any) -> Dict[str, torch.Tensor]:
        tensor = to_tensor(img)
        return {"image": tensor}

    return _transform


def _apply_pipeline(
    inputs: torch.Tensor | Sequence[torch.Tensor],
    pipeline: v2.Transform,
) -> torch.Tensor:
    """Apply a torchvision v2 pipeline per-sample while preserving batch shape."""

    if isinstance(inputs, torch.Tensor):
        if inputs.ndim == 3:
            return pipeline(inputs)
        if inputs.ndim != 4:
            raise ValueError(f"Expected tensor with ndim 3 or 4, got {inputs.ndim}")
        iterable = inputs.unbind(0)
    else:
        iterable = inputs

    outputs = [pipeline(img) for img in iterable]
    if not outputs:
        raise ValueError("_apply_pipeline received an empty batch")
    return torch.stack(outputs, dim=0)


def _maybe_channels_last(tensor: torch.Tensor, channels_last: bool) -> torch.Tensor:
    if channels_last and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def _simclr_pipeline(cfg: AugConfig, size: int) -> v2.Transform:
    mean, std = normalize_stats()
    k = gaussian_kernel_size(size)
    color = v2.ColorJitter(
        brightness=cfg.jitter_strength,
        contrast=cfg.jitter_strength,
        saturation=cfg.jitter_strength,
        hue=0.1,
    )
    transforms: List[v2.Transform] = [
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(
            size,
            scale=(0.08, 1.0),
            ratio=(3 / 4, 4 / 3),
            antialias=True,
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([color], p=0.8),
        v2.RandomGrayscale(p=float(cfg.gray_prob)),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))],
            p=float(cfg.blur_prob),
        ),
    ]
    if float(getattr(cfg, "solarize_prob", 0.0)) > 0.0:
        transforms.append(
            v2.RandomApply(
                [
                    v2.RandomSolarize(
                        threshold=int(getattr(cfg, "solarize_threshold", 128)),
                        additions=0.0,
                    )
                ],
                p=float(cfg.solarize_prob),
            )
        )
    transforms.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(transforms)


def make_simclr_gpu_augment(
    cfg: AugConfig,
    *,
    channels_last: bool = False,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Build a batch-level augmentor that maps 'image' -> ('view1','view2')."""

    pipeline = _simclr_pipeline(cfg, cfg.img_size)

    def _apply_simclr(batch: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        return _apply_pipeline(batch, pipeline)

    def _augment(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base = batch.pop("image", None)
        if base is None:
            return batch
        view1 = _apply_simclr(base)
        view2 = _apply_simclr(base)
        batch["view1"] = _maybe_channels_last(view1, channels_last)
        batch["view2"] = _maybe_channels_last(view2, channels_last)
        return batch

    return _augment


def _swav_pipeline(cfg: AugConfig, size: int, scale: Tuple[float, float]) -> v2.Transform:
    mean, std = normalize_stats()
    k = gaussian_kernel_size(size)
    color = v2.ColorJitter(
        brightness=cfg.jitter_strength,
        contrast=cfg.jitter_strength,
        saturation=cfg.jitter_strength,
        hue=0.1,
    )
    transforms: List[v2.Transform] = [
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size, scale=scale, ratio=(3 / 4, 4 / 3), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([color], p=0.8),
        v2.RandomGrayscale(p=float(cfg.gray_prob)),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))],
            p=float(cfg.blur_prob),
        ),
    ]
    if float(getattr(cfg, "solarize_prob", 0.0)) > 0.0:
        transforms.append(
            v2.RandomApply(
                [
                    v2.RandomSolarize(
                        threshold=int(getattr(cfg, "solarize_threshold", 128)),
                        additions=0.0,
                    )
                ],
                p=float(cfg.solarize_prob),
            )
        )
    transforms.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(transforms)


def make_swav_gpu_augment(
    cfg: AugConfig,
    *,
    channels_last: bool = False,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Build a batch-level augmentor producing multi-crop tensors on device."""

    global_scale = (0.14, 1.0)
    local_scale = (0.05, 0.14)

    global_pipeline = _swav_pipeline(cfg, cfg.img_size, global_scale)

    def _apply_global(batch: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        return _apply_pipeline(batch, global_pipeline)

    local_pipeline: v2.Transform | None = None
    if int(cfg.local_crops) > 0:
        local_pipeline = _swav_pipeline(cfg, cfg.local_size, local_scale)

        def _apply_local(batch: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
            if local_pipeline is None:  # pragma: no cover - defensive
                raise RuntimeError("Local pipeline is not initialized")
            return _apply_pipeline(batch, local_pipeline)
    else:
        _apply_local = None

    def _augment(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base = batch.pop("image", None)
        if base is None:
            return batch
        crops: List[torch.Tensor] = []
        for _ in range(int(cfg.global_crops)):
            crop = _apply_global(base)
            crops.append(_maybe_channels_last(crop, channels_last))
        if int(cfg.local_crops) > 0:
            for _ in range(int(cfg.local_crops)):
                if _apply_local is None:  # pragma: no cover - defensive
                    raise RuntimeError("Local pipeline callable is not available")
                crop = _apply_local(base)
                crops.append(_maybe_channels_last(crop, channels_last))
        batch["crops"] = crops
        batch["code_crops"] = (0, 1)
        return batch

    return _augment


def build_gpu_augmentor(
    method_name: str,
    cfg: AugConfig,
    *,
    channels_last: bool = False,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Return the appropriate GPU augmentor for the given method."""

    method = method_name.lower()
    if method == "swav":
        return make_swav_gpu_augment(cfg, channels_last=channels_last)
    return make_simclr_gpu_augment(cfg, channels_last=channels_last)


__all__ = [
    "build_gpu_pair_pretransform",
    "build_gpu_multicrop_pretransform",
    "build_gpu_augmentor",
    "make_simclr_gpu_augment",
    "make_swav_gpu_augment",
]
