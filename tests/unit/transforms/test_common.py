import numpy as np
from PIL import Image

from mfcl.transforms.common import gaussian_kernel_size, Solarize, to_tensor_and_norm


def test_gaussian_kernel_size():
    k = gaussian_kernel_size(160)
    assert k % 2 == 1 and k >= 3
    assert k == 17


def test_solarize_thresholds(toy_image_rgb):
    img = toy_image_rgb(32)
    inv = Solarize(0)(img)
    same = Solarize(255)(img)
    assert inv.tobytes() != img.tobytes()
    assert same.tobytes() == img.tobytes()


def test_solarize_strict_threshold_behavior():
    arr = np.array([[127, 128, 129]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    sol = Solarize(128)
    out = sol(img)
    assert isinstance(out, Image.Image)
    out_arr = np.array(out)
    assert out_arr[0, 0] == 127  # below threshold unaffected
    assert out_arr[0, 1] == 128  # equal to threshold unaffected
    assert out_arr[0, 2] == 126  # inverted from 129 -> 126


def test_to_tensor_and_norm_returns_transform():
    t = to_tensor_and_norm()
    assert callable(t)
