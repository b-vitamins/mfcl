import contextlib
from typing import List, Tuple

import torch

import mfcl.utils.amp as amp
from mfcl.utils.amp import AmpScaler


def test_amp_scaler_noop_on_cpu() -> None:
    scaler = AmpScaler(enabled=False)
    x = torch.tensor(1.0, requires_grad=True)
    y = scaler.scale(x)
    assert y is x


def test_amp_scaler_step_update(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = AmpScaler(enabled=torch.cuda.is_available())
    with scaler.autocast():
        out = model(torch.ones(1, 2))
        loss = out.sum()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()


def test_amp_scaler_autocast_handles_none_dtype(monkeypatch) -> None:
    calls: List[Tuple[Tuple, dict]] = []

    @contextlib.contextmanager
    def fake_autocast(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    scaler = AmpScaler(enabled=False, amp_dtype=None)
    scaler._enabled = True
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast, raising=True)

    def _raise_legacy(*_args, **_kwargs):  # pragma: no cover - sanity guard
        raise RuntimeError("legacy")

    monkeypatch.setattr(torch.cuda.amp, "autocast", _raise_legacy)

    with scaler.autocast():
        pass

    assert calls == [(("cuda",), {})]


def test_amp_scaler_state_roundtrip_handles_disabled() -> None:
    scaler = AmpScaler(enabled=False)
    assert scaler.state_dict() == {}
    scaler.load_state_dict({})
    scaler.load_state_dict({"foo": 1})


def test_amp_scaler_state_roundtrip_enabled() -> None:
    scaler = AmpScaler(enabled=False)
    scaler._enabled = True

    class _FakeScaler:
        def __init__(self) -> None:
            self.loaded = None

        def state_dict(self):
            return {"foo": "bar"}

        def load_state_dict(self, state):
            self.loaded = state

    fake = _FakeScaler()
    scaler._scaler = fake

    assert scaler.state_dict() == {"foo": "bar"}
    scaler.load_state_dict({"foo": "baz"})
    assert fake.loaded == {"foo": "baz"}


def test_amp_scaler_disables_when_cuda_unavailable(monkeypatch, caplog) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    amp._WARNED_NO_CUDA = False
    with caplog.at_level("WARNING"):
        scaler = AmpScaler(enabled=True)
    assert not scaler.is_enabled
    assert any(
        "AMP requested but CUDA is unavailable" in record.getMessage()
        for record in caplog.records
    )
