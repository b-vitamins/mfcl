import pytest
import torch

from mfcl.data.schema import extract_labels


def test_extract_labels_returns_tensor():
    labels = torch.arange(4)
    batch = {"target": labels}
    out = extract_labels(batch)
    assert out is labels


@pytest.mark.parametrize("legacy_key", ["labels", "label", "mixture_labels"])
def test_extract_labels_rejects_legacy_keys(legacy_key: str):
    batch = {legacy_key: torch.arange(3)}
    with pytest.raises(KeyError) as exc:
        extract_labels(batch)
    assert "canonical key" in str(exc.value)


def test_extract_labels_required_missing():
    with pytest.raises(KeyError) as exc:
        extract_labels({}, required=True)
    assert "missing required" in str(exc.value)


def test_extract_labels_required_none():
    batch = {"target": None}
    with pytest.raises(ValueError):
        extract_labels(batch, required=True)


def test_extract_labels_rejects_non_tensor():
    batch = {"target": [1, 2, 3]}
    with pytest.raises(TypeError):
        extract_labels(batch)
