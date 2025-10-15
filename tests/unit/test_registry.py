import pytest

from mfcl.core.registry import Registry


def test_registry_add_get_has_keys():
    r = Registry("encoders")
    r.add("resnet18", object)
    assert r.has("resnet18")
    assert r.get("resnet18") is object
    assert "resnet18" in r.keys()


def test_registry_duplicate_key():
    r = Registry("x")
    r.add("a", object)
    with pytest.raises(KeyError):
        r.add("a", object)


@pytest.mark.parametrize("bad", ["ResNet18", "resnet 18", "resnet_18", "", "resnet@18"])
def test_registry_bad_keys(bad):
    r = Registry("x")
    with pytest.raises((ValueError, TypeError)):
        r.add(bad, object)


def test_registry_strict_overwrite_opt_in():
    r = Registry("x")
    r.add("a", int)
    r.add("a", float, strict=False)
    assert r.get("a") is float


def test_registry_dunder_helpers():
    r = Registry("debug")
    r.add("a", int)
    assert "a" in r
    assert len(r) == 1
    rep = repr(r)
    assert "Registry" in rep and "keys=1" in rep
