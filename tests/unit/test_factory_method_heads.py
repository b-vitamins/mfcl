from __future__ import annotations

import pytest

import mfcl.core.factory as F
from mfcl.core.config import (
    AugConfig,
    Config,
    DataConfig,
    MethodConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from tests.helpers.nets import TinyEncoder


def _make_cfg(method_name: str, **method_kwargs: object) -> Config:
    return Config(
        data=DataConfig(root="/tmp"),
        aug=AugConfig(img_size=64),
        model=ModelConfig(
            encoder="resnet18",
            encoder_dim=16,
            projector_hidden=32,
            projector_out=8,
            predictor_hidden=16,
            predictor_out=8,
        ),
        method=MethodConfig(name=method_name, **method_kwargs),
        optim=OptimConfig(),
        train=TrainConfig(),
    )


@pytest.mark.parametrize(
    "method_name, projector_attrs, predictor_attrs, method_kwargs",
    [
        ("simclr", ("projector",), (), {}),
        ("moco", ("projector_q", "projector_k"), (), {"moco_queue": 16}),
        (
            "byol",
            ("g_q", "g_k"),
            ("q",),
            {"byol_momentum_schedule": "const"},
        ),
        ("simsiam", ("projector",), ("predictor",), {}),
        ("barlow", ("projector",), (), {}),
        ("vicreg", ("projector",), (), {}),
        (
            "swav",
            ("projector",),
            (),
            {"swav_prototypes": 8, "swav_codes_queue_size": 0},
        ),
    ],
)
def test_build_method_wires_registered_heads(
    monkeypatch,
    method_name: str,
    projector_attrs: tuple[str, ...],
    predictor_attrs: tuple[str, ...],
    method_kwargs: dict[str, object],
) -> None:
    monkeypatch.setattr(
        F,
        "build_encoder",
        lambda cfg: TinyEncoder(cfg.model.encoder_dim),
    )
    cfg = _make_cfg(method_name, **method_kwargs)

    method = F.build_method(cfg)

    projector_ctor = F.HEAD_REGISTRY.get("projector")
    projectors = [getattr(method, attr) for attr in projector_attrs]
    for head in projectors:
        assert isinstance(head, projector_ctor)
    if len(projectors) > 1:
        assert len({id(head) for head in projectors}) == len(projectors)

    if predictor_attrs:
        predictor_ctor = F.HEAD_REGISTRY.get("predictor")
        for attr in predictor_attrs:
            head = getattr(method, attr)
            assert isinstance(head, predictor_ctor)
