from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mfcl.utils.seed import set_seed


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_global_seed(request):
    # Derive a unique, stable seed per test nodeid
    base = 1337
    sid = abs(hash(request.node.nodeid)) % (2**31)
    seed = (base + sid) % (2**31 - 1)
    set_seed(seed, deterministic=True)
    yield


@pytest.fixture()
def tmp_runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def toy_image_rgb():
    def _make(size: int = 64) -> Image.Image:
        # Deterministic RGB gradient image
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            arr[..., c] = np.linspace(0, 255, num=size, dtype=np.uint8)[None, :]
        return Image.fromarray(arr, mode="RGB")

    return _make


@pytest.fixture()
def toy_ssl_batch_pair(device):
    def _make(B: int = 4, C: int = 3, H: int = 64):
        x1 = torch.randn(B, C, H, H, device=device)
        x2 = torch.randn(B, C, H, H, device=device)
        return {"view1": x1, "view2": x2, "index": torch.arange(B, device=device)}

    return _make


@pytest.fixture()
def toy_ssl_batch_multicrop(device):
    def _make(B: int = 2, C: int = 3, Hg: int = 64, Hl: int = 32, locals_n: int = 2):
        crops = [torch.randn(B, C, Hg, Hg, device=device) for _ in range(2)]
        crops += [torch.randn(B, C, Hl, Hl, device=device) for _ in range(locals_n)]
        return {
            "crops": crops,
            "code_crops": (0, 1),
            "index": torch.arange(B, device=device),
        }

    return _make


@pytest.fixture()
def dummy_encoder():
    from tests.helpers.nets import TinyEncoder

    def _make(D_out: int = 32):
        return TinyEncoder(D_out)

    return _make


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA")
    config.addinivalue_line(
        "markers", "slow: longer-running integration or property tests"
    )
    config.addinivalue_line("markers", "dist: uses torch.distributed")
