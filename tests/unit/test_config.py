import pytest

from mfcl.core.config import (
    DataConfig,
    AugConfig,
    ModelConfig,
    MethodConfig,
    OptimConfig,
    TrainConfig,
    Config,
    to_omegaconf,
    from_omegaconf,
    validate,
)


def test_config_roundtrip_and_validate():
    cfg = Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(),
        model=ModelConfig(),
        method=MethodConfig(name="simclr"),
        optim=OptimConfig(),
        train=TrainConfig(),
    )
    oc = to_omegaconf(cfg)
    cfg2 = from_omegaconf(oc)
    assert cfg2.data.root == "/tmp"
    validate(cfg2)


def test_validation_errors_reference_fields():
    cfg = Config(
        data=DataConfig(root="/tmp", batch_size=0),
        aug=AugConfig(img_size=32),
        model=ModelConfig(projector_out=0),
        method=MethodConfig(name="moco", moco_queue=1),
        optim=OptimConfig(lr=0.0),
        train=TrainConfig(epochs=0),
    )
    with pytest.raises(ValueError) as e:
        validate(cfg)
    msg = str(e.value)
    assert any(
        k in msg
        for k in [
            "batch_size",
            "img_size",
            "projector_out",
            "moco_queue",
            "lr",
            "epochs",
        ]
    )
