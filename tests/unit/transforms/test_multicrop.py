import pytest

from mfcl.core.config import AugConfig
from mfcl.transforms.common import _find_solarize_in_compose
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


def test_multicrop_requires_two_global_crops():
    cfg = AugConfig(
        img_size=64,
        global_crops=1,
        local_crops=0,
        local_size=32,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=0.0,
    )
    with pytest.raises(ValueError):
        build_multicrop_transforms(cfg)


def test_multicrop_handles_zero_local_crops(toy_image_rgb):
    cfg = AugConfig(
        img_size=64,
        local_crops=0,
        local_size=32,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=0.0,
    )
    tf = build_multicrop_transforms(cfg)
    out = tf(toy_image_rgb(128))
    assert out["code_crops"] == (0, 1)
    assert len(out["crops"]) == 2
    closure = tf.__closure__
    assert closure is not None and len(closure) >= 3
    # Local chain should be absent in closure
    assert closure[2].cell_contents is None


def test_multicrop_uses_solarize_threshold():
    cfg = AugConfig(
        img_size=64,
        global_crops=2,
        local_crops=1,
        local_size=32,
        jitter_strength=0.0,
        blur_prob=0.0,
        gray_prob=0.0,
        solarize_prob=1.0,
    )
    cfg.solarize_threshold = 21
    tf = build_multicrop_transforms(cfg)
    closure = tf.__closure__
    assert closure is not None and len(closure) >= 3
    g_chain = closure[1].cell_contents
    l_chain = closure[2].cell_contents
    g_solar = _find_solarize_in_compose(g_chain)
    l_solar = _find_solarize_in_compose(l_chain)
    assert g_solar and g_solar[0].threshold == 21
    assert l_solar and l_solar[0].threshold == 21
