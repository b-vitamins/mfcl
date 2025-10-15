import torch

from mfcl.utils.amp import AmpScaler


def test_amp_scaler_noop_on_cpu():
    scaler = AmpScaler(enabled=False)
    x = torch.tensor(1.0, requires_grad=True)
    y = scaler.scale(x)
    assert y is x


def test_amp_scaler_step_update(tmp_path):
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


def test_amp_scaler_state_roundtrip_handles_disabled():
    scaler = AmpScaler(enabled=False)
    assert scaler.state_dict() == {}
    # Loading any state should be a no-op when AMP is disabled
    scaler.load_state_dict({})
    scaler.load_state_dict({"foo": 1})


def test_amp_scaler_state_roundtrip_enabled():
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
