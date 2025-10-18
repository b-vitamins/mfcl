import torch

from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from mfcl.core.factory import build_sched


def _make_cfg(*, step_on: str, cosine: bool, warmup: int, epochs: int) -> Config:
    return Config(
        data=DataConfig(root="/tmp", name="synthetic"),
        aug=AugConfig(),
        model=ModelConfig(),
        method=MethodConfig(),
        optim=OptimConfig(),
        train=TrainConfig(
            epochs=epochs,
            warmup_epochs=warmup,
            cosine=cosine,
            scheduler_step_on=step_on,
            amp=False,
            save_dir="",
        ),
    )


def test_build_sched_batch_mode_uses_step_horizons():
    cfg = _make_cfg(step_on="batch", cosine=True, warmup=2, epochs=6)
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = build_sched(cfg, opt, steps_per_epoch=5)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)
    warmup, after = sched._schedulers  # type: ignore[attr-defined]
    assert warmup.total_iters == 10
    assert isinstance(after, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert after.T_max == 20


def test_build_sched_epoch_mode_uses_epoch_horizons():
    cfg = _make_cfg(step_on="epoch", cosine=False, warmup=1, epochs=5)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = build_sched(cfg, opt, steps_per_epoch=7)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)
    warmup, after = sched._schedulers  # type: ignore[attr-defined]
    assert warmup.total_iters == 1
    assert isinstance(after, torch.optim.lr_scheduler.StepLR)
    assert after.step_size == int(0.67 * (5 - 1))
