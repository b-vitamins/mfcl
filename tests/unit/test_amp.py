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
