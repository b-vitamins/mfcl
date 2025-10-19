import pytest
import torch

from mfcl.core.config import AugConfig
from mfcl.transforms.gpu import (
    build_gpu_augmentor,
    build_gpu_multicrop_pretransform,
    build_gpu_pair_pretransform,
)

try:
    import torchvision.transforms.v2  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    pytest.skip("torchvision.transforms.v2 is required for GPU augmentation tests", allow_module_level=True)


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
def test_pair_gpu_augment_shapes(device, channels_last):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = AugConfig(backend="tv2")
    pre = build_gpu_pair_pretransform(cfg)

    from PIL import Image

    img = Image.fromarray((torch.rand(3, cfg.img_size, cfg.img_size) * 255).byte().permute(1, 2, 0).numpy())
    sample = pre(img)
    assert "image" in sample
    batch = {
        "image": sample["image"].unsqueeze(0).to(device),
        "index": torch.tensor([0], device=device),
    }

    augmentor = build_gpu_augmentor("simclr", cfg, channels_last=channels_last)
    batch = augmentor(batch)
    v1 = batch["view1"]
    v2 = batch["view2"]
    assert v1.shape == (1, 3, cfg.img_size, cfg.img_size)
    assert v2.shape == (1, 3, cfg.img_size, cfg.img_size)
    assert v1.device == device
    assert v1.dtype == torch.float32
    if channels_last and device.type == "cuda":
        assert v1.is_contiguous(memory_format=torch.channels_last)
        assert v2.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
def test_multicrop_gpu_augment_returns_crops(device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = AugConfig(local_crops=2, local_size=96, backend="tv2")
    pre = build_gpu_multicrop_pretransform(cfg)

    from PIL import Image

    img = Image.fromarray((torch.rand(3, cfg.img_size, cfg.img_size) * 255).byte().permute(1, 2, 0).numpy())
    sample = pre(img)
    batch = {
        "image": sample["image"].unsqueeze(0).to(device),
        "index": torch.tensor([0], device=device),
    }
    augmentor = build_gpu_augmentor("swav", cfg, channels_last=False)
    batch = augmentor(batch)
    crops = batch["crops"]
    assert isinstance(crops, list)
    expected = cfg.global_crops + cfg.local_crops
    assert len(crops) == expected
    for idx, crop in enumerate(crops):
        assert crop.shape[0] == 1
        if idx < cfg.global_crops:
            assert crop.shape[-2:] == (cfg.img_size, cfg.img_size)
        else:
            assert crop.shape[-2:] == (cfg.local_size, cfg.local_size)
        assert crop.device == device
        assert crop.dtype == torch.float32
    assert batch["code_crops"] == (0, 1)
