"""Typed configuration dataclasses and OmegaConf glue.

This module defines the configuration structure for data, augmentation, model,
method, optimization, and training. It also provides helpers to convert to and
from OmegaConf without requiring Hydra at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class DataConfig:
    """Dataset and dataloader options.

    Note:
        ``persistent_workers`` is ignored when ``num_workers`` is zero because
        PyTorch disallows that combination. ``factory.build_data`` enforces the
        safe behavior at runtime.
    """

    root: str
    name: str = "imagenet"
    train_list: str | None = None
    val_list: str | None = None
    num_workers: int = 8
    batch_size: int = 256
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    synthetic_num_classes: int = 1000
    synthetic_train_size: int = 2048
    synthetic_val_size: int = 512


@dataclass
class AugConfig:
    """Augmentation parameters for self-supervised methods."""

    img_size: int = 160
    global_crops: int = 2
    local_crops: int = 0
    local_size: int = 96
    jitter_strength: float = 0.4
    blur_prob: float = 0.5
    gray_prob: float = 0.2
    solarize_prob: float = 0.0
    solarize_threshold: int = 128


@dataclass
class ModelConfig:
    """Backbone and head dimensions/options."""

    encoder: str = "resnet18"
    encoder_dim: int = 512
    projector_hidden: int = 2048
    projector_out: int = 128
    predictor_hidden: int = 1024
    predictor_out: int = 128
    norm_feat: bool = True


@dataclass
class MethodConfig:
    """Method-specific hyperparameters."""

    name: str = "simclr"  # simclr|moco|byol|simsiam|swav|barlow|vicreg
    temperature: float = 0.1
    moco_momentum: float = 0.999
    moco_queue: int = 65536
    byol_tau_base: float = 0.996
    swav_prototypes: int = 3000
    swav_sinkhorn_iters: int = 3
    barlow_lambda: float = 5e-3
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0


@dataclass
class OptimConfig:
    """Optimizer options."""

    name: str = "sgd"  # sgd|lars|adamw
    lr: float = 0.05
    weight_decay: float = 1e-4
    momentum: float = 0.9  # used by sgd/lars
    betas: Tuple[float, float] = (0.9, 0.999)  # adamw


@dataclass
class TrainConfig:
    """Training loop options."""

    epochs: int = 200
    warmup_epochs: int = 10
    cosine: bool = True
    amp: bool = True
    grad_clip: float | None = None
    save_dir: str = "runs/simclr_r18_160"
    seed: int = 42
    log_interval: int = 1  # batches
    # Reserved for trainer control; factory is agnostic to step granularity.
    scheduler_step_on: str = "epoch"


@dataclass
class Config:
    """Aggregate configuration node encompassing all sections."""

    data: DataConfig
    aug: AugConfig
    model: ModelConfig
    method: MethodConfig
    optim: OptimConfig
    train: TrainConfig


def validate(cfg: Config) -> None:
    """Validate a configuration object for logical consistency.

    Args:
        cfg: Config to validate.

    Raises:
        ValueError: If any constraint is violated. Error messages include the
            offending field name for quick remediation.
    """
    dataset_kind = getattr(cfg.data, "name", "imagenet").lower()
    if dataset_kind not in {"imagenet", "synthetic"}:
        raise ValueError("data.name must be 'imagenet' or 'synthetic'")

    if cfg.data.batch_size <= 0:
        raise ValueError("data.batch_size must be > 0")
    if cfg.data.num_workers < 0:
        raise ValueError("data.num_workers must be >= 0")
    if cfg.aug.img_size < 64:
        raise ValueError("aug.img_size must be >= 64")

    if dataset_kind == "synthetic":
        if cfg.data.synthetic_num_classes <= 0:
            raise ValueError("data.synthetic_num_classes must be > 0 for synthetic data")
        if cfg.data.synthetic_train_size <= 0:
            raise ValueError("data.synthetic_train_size must be > 0 for synthetic data")
        if cfg.data.synthetic_val_size <= 0:
            raise ValueError("data.synthetic_val_size must be > 0 for synthetic data")

    if cfg.model.encoder_dim <= 0:
        raise ValueError("model.encoder_dim must be > 0")
    if cfg.model.projector_out <= 0:
        raise ValueError("model.projector_out must be > 0")
    if cfg.optim.lr <= 0:
        raise ValueError("optim.lr must be > 0")
    if cfg.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0")
    if cfg.train.grad_clip is not None and cfg.train.grad_clip <= 0:
        raise ValueError("train.grad_clip must be > 0 when set")

    if hasattr(cfg.train, "scheduler_step_on"):
        if cfg.train.scheduler_step_on not in {"batch", "epoch"}:
            raise ValueError("train.scheduler_step_on must be 'batch' or 'epoch'")

    valid_methods = {"simclr", "moco", "byol", "simsiam", "swav", "barlow", "vicreg"}
    if cfg.method.name not in valid_methods:
        raise ValueError("method.name must be one of simclr|moco|byol|simsiam|swav|barlow|vicreg")

    if cfg.method.name == "swav" and cfg.aug.global_crops < 2:
        raise ValueError("aug.global_crops must be >= 2 for method.swav")
    if cfg.method.name != "swav" and cfg.aug.local_crops > 0:
        raise ValueError("aug.local_crops must be 0 unless method.swav is used")

    if cfg.aug.local_crops > 0:
        if not hasattr(cfg.aug, "local_size"):
            raise ValueError("aug.local_size must be set when aug.local_crops > 0")
        if cfg.aug.local_size < 64:
            raise ValueError("aug.local_size must be >= 64 when aug.local_crops > 0")

    if cfg.method.name in {"simclr", "moco", "swav"} and cfg.method.temperature <= 0:
        raise ValueError("method.temperature must be > 0 for simclr/moco/swav")
    if cfg.method.name == "moco" and cfg.method.moco_queue < 1024:
        raise ValueError("method.moco_queue must be >= 1024 for method.moco")
    if cfg.method.name == "barlow" and cfg.method.barlow_lambda <= 0:
        raise ValueError("method.barlow_lambda must be > 0 for method.barlow")
    if cfg.method.name in {"byol", "simsiam"}:
        if cfg.model.predictor_out != cfg.model.projector_out:
            raise ValueError(
                "model.predictor_out must match model.projector_out for byol/simsiam"
            )
    if cfg.method.name == "vicreg":
        if cfg.method.vicreg_lambda <= 0:
            raise ValueError("method.vicreg_lambda must be > 0 for method.vicreg")
        if cfg.method.vicreg_mu <= 0:
            raise ValueError("method.vicreg_mu must be > 0 for method.vicreg")
        if cfg.method.vicreg_nu <= 0:
            raise ValueError("method.vicreg_nu must be > 0 for method.vicreg")


def to_omegaconf(cfg: Config) -> Any:
    """Convert a dataclass-based config to an OmegaConf tree.

    Import of OmegaConf is local to avoid import-time dependency on Hydra.

    Args:
        cfg: Dataclass configuration instance.

    Returns:
        An OmegaConf configuration object mirroring the dataclass structure.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        # Degrade gracefully when OmegaConf is not available by returning
        # a plain dict that from_omegaconf can consume.
        return asdict(cfg)

    try:
        return OmegaConf.create(asdict(cfg))
    except Exception as exc:
        raise RuntimeError(
            "OmegaConf.create failed while converting Config: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc


def _to_plain_dict(oc: Any) -> Dict[str, Any]:
    """Convert an OmegaConf or mapping-like object to a plain dict.

    Uses DictConfig/ListConfig isinstance checks when OmegaConf is available;
    otherwise falls back to a best-effort ``dict(oc)`` cast.
    """
    try:
        # Attempt OmegaConf conversion path with explicit config types.
        from omegaconf import OmegaConf, DictConfig, ListConfig  # type: ignore

        if isinstance(oc, (DictConfig, ListConfig)):
            return OmegaConf.to_container(oc, resolve=True)  # type: ignore
    except ImportError:
        pass

    if isinstance(oc, Mapping):
        return dict(oc)
    raise TypeError(
        f"from_omegaconf expects a mapping-like object, got {type(oc).__name__}"
    )


def from_omegaconf(oc: Any) -> Config:
    """Convert an OmegaConf node-tree back to a typed :class:`Config`.

    Args:
        oc: OmegaConf configuration or a mapping-equivalent.

    Returns:
        A fully typed :class:`Config` instance.

    Raises:
        ValueError: If required sections or fields are missing.
    """
    d = _to_plain_dict(oc)

    def req(section: str) -> Dict[str, Any]:
        if section not in d:
            raise ValueError(f"Missing required config section: '{section}'")
        v = d[section]
        if not isinstance(v, dict):
            raise ValueError(f"Section '{section}' must be a mapping.")
        return v

    data = req("data")
    aug = req("aug")
    model = req("model")
    method = req("method")
    optim = req("optim")
    train = req("train")

    data_cfg = DataConfig(**data)
    aug_cfg = AugConfig(**aug)
    model_cfg = ModelConfig(**model)
    method_cfg = MethodConfig(**method)
    optim_cfg = OptimConfig(**optim)
    train_cfg = TrainConfig(**train)

    cfg = Config(
        data=data_cfg,
        aug=aug_cfg,
        model=model_cfg,
        method=method_cfg,
        optim=optim_cfg,
        train=train_cfg,
    )
    validate(cfg)
    return cfg


__all__ = [
    "DataConfig",
    "AugConfig",
    "ModelConfig",
    "MethodConfig",
    "OptimConfig",
    "TrainConfig",
    "Config",
    "validate",
    "to_omegaconf",
    "from_omegaconf",
]
