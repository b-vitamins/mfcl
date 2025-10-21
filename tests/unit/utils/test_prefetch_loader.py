import pytest
import torch

from mfcl.utils.prefetch import PrefetchLoader


def _dummy_loader(num_batches: int):
    for _ in range(num_batches):
        yield {"input": torch.randn(2, 3, 4)}


def test_prefetch_loader_depth_consumes_all():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to test PrefetchLoader")
    device = torch.device("cuda")
    loader = _dummy_loader(3)
    prefetch = PrefetchLoader(loader, device, prefetch_depth=2)
    batches = list(prefetch)
    assert len(batches) == 3
    for batch in batches:
        assert batch["input"].device == device
        assert batch["input"].dtype == torch.float32


def test_prefetch_loader_context_manager_cleans_up_streams():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to test PrefetchLoader")
    device = torch.device("cuda")
    loader = _dummy_loader(2)
    with PrefetchLoader(loader, device, prefetch_depth=2) as prefetch:
        iterator = iter(prefetch)
        next(iterator)
    assert not prefetch._streams
    assert not prefetch._queue
    assert prefetch._iterator is None


def test_prefetch_loader_releases_streams_after_exhaustion():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to test PrefetchLoader")
    device = torch.device("cuda")
    loader = _dummy_loader(2)
    prefetch = PrefetchLoader(loader, device, prefetch_depth=2)
    iterator = iter(prefetch)
    with pytest.raises(StopIteration):
        while True:
            next(iterator)
    assert not prefetch._streams
    assert not prefetch._queue
