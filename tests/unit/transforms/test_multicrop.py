from mfcl.core.config import AugConfig
from mfcl.transforms.multicrop import build_multicrop_transforms


def test_multicrop_counts_and_code_crops(toy_image_rgb):
    cfg = AugConfig(
        img_size=64,
        local_crops=2,
        local_size=32,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=0.0,
    )
    tf = build_multicrop_transforms(cfg)
    out = tf(toy_image_rgb(128))
    assert out["code_crops"] == (0, 1)
    crops = out["crops"]
    assert len(crops) == 4
    assert crops[0].shape[-2:] == (64, 64)
    assert crops[-1].shape[-2:] == (32, 32)
