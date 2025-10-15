from mfcl.core.config import AugConfig
from mfcl.transforms.common import _find_solarize_in_compose
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


def test_pair_transforms_respects_solarize_threshold():
    cfg = AugConfig(
        img_size=64,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=1.0,
    )
    cfg.solarize_threshold = 37
    tf = build_pair_transforms(cfg)
    compose = tf.__closure__[0].cell_contents
    found = _find_solarize_in_compose(compose)
    assert found and found[0].threshold == 37
