from mfcl.core.config import AugConfig
from mfcl.transforms.simclr import build_pair_transforms


def test_pair_transforms_keys_and_sizes(toy_image_rgb):
    cfg = AugConfig(
        img_size=64,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=0.0,
    )
    tf = build_pair_transforms(cfg)
    out = tf(toy_image_rgb(128))
    assert set(out.keys()) == {"view1", "view2"}
    assert out["view1"].shape == out["view2"].shape
