from mfcl.transforms.common import gaussian_kernel_size, Solarize, to_tensor_and_norm


def test_gaussian_kernel_size():
    k = gaussian_kernel_size(160)
    assert k % 2 == 1 and k >= 3


def test_solarize_thresholds(toy_image_rgb):
    img = toy_image_rgb(32)
    inv = Solarize(0)(img)
    same = Solarize(255)(img)
    assert inv.tobytes() != img.tobytes()
    assert same.tobytes() == img.tobytes()


def test_to_tensor_and_norm_returns_transform():
    t = to_tensor_and_norm()
    assert callable(t)
