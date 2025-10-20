"""Typed configuration dataclasses and OmegaConf glue.

This module defines the configuration structure for data, augmentation, model,
method, optimization, and training. It also provides helpers to convert to and
from OmegaConf without requiring Hydra at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class ClassPackedConfig:
    """Options for class-balanced batch sampling."""

    enabled: bool = False
    num_classes_per_batch: int = 4
    instances_per_class: int = 2
    seed: int = 0


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
    prefetch_factor: int = 2
    multiprocessing_context: str | None = None
    worker_threads: int = 1
    pin_memory_device: str | None = "cuda"
    drop_last: bool = True
    synthetic_num_classes: int = 1000
    synthetic_train_size: int = 2048
    synthetic_val_size: int = 512
    class_packed: ClassPackedConfig = field(default_factory=ClassPackedConfig)


@dataclass
class SyntheticClustersConfig:
    """Configuration for synthetic clustered embedding generation."""

    enabled: bool = False
    k: int = 8
    scale: float = 0.2
    samples: int = 8192
    dim: int = 128
    seed: int = 0
    output_dir: str | None = None


@dataclass
class HeavyTailConfig:
    """Configuration for heavy-tail similarity injection diagnostics."""

    enabled: bool = False
    p_tail: float = 0.1
    tail_scale: float = 5.0


@dataclass
class StressConfig:
    """Aggregate stress-testing diagnostics toggles."""

    synthetic_clusters: SyntheticClustersConfig = field(default_factory=SyntheticClustersConfig)
    heavy_tail: HeavyTailConfig = field(default_factory=HeavyTailConfig)


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
    backend: str = "cpu"


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
    moco_queue_device: str = "cpu"
    moco_queue_dtype: str = "fp32"
    byol_tau_base: float = 0.996
    byol_tau_final: float = 0.996
    byol_momentum_schedule: str = "const"
    byol_momentum_schedule_steps: int | None = None
    ntxent_mode: str = "paired"
    use_syncbn: bool = False
    cross_rank_negatives: bool = False
    cross_rank_queue: bool = False
    swav_prototypes: int = 3000
    swav_sinkhorn_iters: int = 3
    swav_sinkhorn_tol: float = 1e-3
    swav_sinkhorn_max_iters: int = 100
    swav_use_fp32_sinkhorn: bool = True
    swav_codes_queue_size: int = 0
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
    log_interval: int = 50  # batches
    channels_last: bool = False
    prefetch_gpu: bool = False
    gpu_augment: bool = False
    prefetch_depth: int = 1
    loss_fp32: bool = True
    cudnn_bench: bool = True
    compile: bool = False
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
    stress: StressConfig = field(default_factory=StressConfig)


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
    if cfg.data.prefetch_factor <= 0:
        raise ValueError("data.prefetch_factor must be > 0")
    if cfg.data.worker_threads <= 0:
        raise ValueError("data.worker_threads must be > 0")
    pin_mem_dev = getattr(cfg.data, "pin_memory_device", None)
    if pin_mem_dev is not None and not isinstance(pin_mem_dev, str):
        raise ValueError("data.pin_memory_device must be a string or null")
    if cfg.aug.img_size < 64:
        raise ValueError("aug.img_size must be >= 64")

    aug_backend = getattr(cfg.aug, "backend", "cpu").lower()
    if aug_backend not in {"cpu", "tv2"}:
        raise ValueError("aug.backend must be 'cpu' or 'tv2'")

    if dataset_kind == "synthetic":
        if cfg.data.synthetic_num_classes <= 0:
            raise ValueError("data.synthetic_num_classes must be > 0 for synthetic data")
        if cfg.data.synthetic_train_size <= 0:
            raise ValueError("data.synthetic_train_size must be > 0 for synthetic data")
        if cfg.data.synthetic_val_size <= 0:
            raise ValueError("data.synthetic_val_size must be > 0 for synthetic data")

    class_cfg = getattr(cfg.data, "class_packed", None)
    if class_cfg is not None:
        if class_cfg.num_classes_per_batch <= 0:
            raise ValueError("data.class_packed.num_classes_per_batch must be > 0")
        if class_cfg.instances_per_class <= 0:
            raise ValueError("data.class_packed.instances_per_class must be > 0")
        if class_cfg.seed < 0:
            raise ValueError("data.class_packed.seed must be >= 0")

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
    if getattr(cfg.train, "prefetch_depth", 1) <= 0:
        raise ValueError("train.prefetch_depth must be > 0")
    if getattr(cfg.train, "gpu_augment", False) not in {True, False}:
        raise ValueError("train.gpu_augment must be boolean")
    if getattr(cfg.train, "loss_fp32", True) not in {True, False}:
        raise ValueError("train.loss_fp32 must be boolean")
    if getattr(cfg.train, "cudnn_bench", True) not in {True, False}:
        raise ValueError("train.cudnn_bench must be boolean")

    if hasattr(cfg.train, "scheduler_step_on"):
        if cfg.train.scheduler_step_on not in {"batch", "epoch"}:
            raise ValueError("train.scheduler_step_on must be 'batch' or 'epoch'")

    if aug_backend != "cpu" and not getattr(cfg.train, "gpu_augment", False):
        raise ValueError("train.gpu_augment must be true when aug.backend != 'cpu'")

    valid_methods = {"simclr", "moco", "byol", "simsiam", "swav", "barlow", "vicreg"}
    if cfg.method.name not in valid_methods:
        raise ValueError("method.name must be one of simclr|moco|byol|simsiam|swav|barlow|vicreg")

    method_name = cfg.method.name.lower()
    queue_device = getattr(cfg.method, "moco_queue_device", "cpu").lower()
    if queue_device not in {"cpu", "cuda"}:
        raise ValueError("method.moco_queue_device must be 'cpu' or 'cuda'")
    queue_dtype = getattr(cfg.method, "moco_queue_dtype", "fp32").lower()
    if queue_dtype not in {"fp32", "float32", "fp16", "float16"}:
        raise ValueError("method.moco_queue_dtype must be fp32/float32 or fp16/float16")
    if method_name == "swav" and cfg.aug.global_crops < 2:
        raise ValueError("aug.global_crops must be >= 2 for method.swav")
    if method_name != "swav" and cfg.aug.local_crops > 0:
        raise ValueError("aug.local_crops must be 0 unless method.swav is used")

    if cfg.aug.local_crops > 0:
        if not hasattr(cfg.aug, "local_size"):
            raise ValueError("aug.local_size must be set when aug.local_crops > 0")
        if cfg.aug.local_size < 64:
            raise ValueError("aug.local_size must be >= 64 when aug.local_crops > 0")

    ntxent_mode = getattr(cfg.method, "ntxent_mode", "paired")
    if ntxent_mode.lower() not in {"paired", "2n", "twon"}:
        raise ValueError("method.ntxent_mode must be 'paired' or '2N'")

    if method_name in {"simclr", "moco", "swav"} and cfg.method.temperature <= 0:
        raise ValueError("method.temperature must be > 0 for simclr/moco/swav")
    if method_name == "moco" and cfg.method.moco_queue < 1024:
        raise ValueError("method.moco_queue must be >= 1024 for method.moco")
    if method_name == "moco":
        if cfg.method.cross_rank_queue not in {True, False}:
            raise ValueError("method.cross_rank_queue must be boolean")
        if cfg.method.use_syncbn not in {True, False}:
            raise ValueError("method.use_syncbn must be boolean")
    if cfg.method.cross_rank_negatives not in {True, False}:
        raise ValueError("method.cross_rank_negatives must be boolean")
    if cfg.method.swav_use_fp32_sinkhorn not in {True, False}:
        raise ValueError("method.swav_use_fp32_sinkhorn must be boolean")

    if cfg.method.swav_sinkhorn_tol <= 0:
        raise ValueError("method.swav_sinkhorn_tol must be > 0")
    if cfg.method.swav_sinkhorn_max_iters < 1:
        raise ValueError("method.swav_sinkhorn_max_iters must be >= 1")
    if cfg.method.swav_codes_queue_size < 0:
        raise ValueError("method.swav_codes_queue_size must be >= 0")

    tau_base = getattr(cfg.method, "byol_tau_base", 0.996)
    tau_final = getattr(cfg.method, "byol_tau_final", tau_base)
    if not 0.0 <= tau_base < 1.0:
        raise ValueError("method.byol_tau_base must be in [0, 1)")
    if not 0.0 <= tau_final < 1.0:
        raise ValueError("method.byol_tau_final must be in [0, 1)")
    schedule = getattr(cfg.method, "byol_momentum_schedule", "const").lower()
    if schedule not in {"const", "cosine"}:
        raise ValueError("method.byol_momentum_schedule must be 'const' or 'cosine'")
    steps = cfg.method.byol_momentum_schedule_steps
    if schedule == "cosine" and steps is not None and steps <= 0:
        raise ValueError("method.byol_momentum_schedule_steps must be > 0 for cosine schedule")

    if method_name == "barlow" and cfg.method.barlow_lambda <= 0:
        raise ValueError("method.barlow_lambda must be > 0 for method.barlow")
    if method_name in {"byol", "simsiam"}:
        if cfg.model.predictor_out != cfg.model.projector_out:
            raise ValueError(
                "model.predictor_out must match model.projector_out for byol/simsiam"
            )
    if method_name == "vicreg":
        if cfg.method.vicreg_lambda <= 0:
            raise ValueError("method.vicreg_lambda must be > 0 for method.vicreg")
        if cfg.method.vicreg_mu <= 0:
            raise ValueError("method.vicreg_mu must be > 0 for method.vicreg")
        if cfg.method.vicreg_nu <= 0:
            raise ValueError("method.vicreg_nu must be > 0 for method.vicreg")

    stress_cfg = getattr(cfg, "stress", None)
    if stress_cfg is not None:
        synth_cfg = getattr(stress_cfg, "synthetic_clusters", None)
        if synth_cfg is not None:
            if synth_cfg.k <= 0:
                raise ValueError("stress.synthetic_clusters.k must be > 0")
            if synth_cfg.scale <= 0:
                raise ValueError("stress.synthetic_clusters.scale must be > 0")
            if synth_cfg.samples <= 0:
                raise ValueError("stress.synthetic_clusters.samples must be > 0")
            if synth_cfg.dim <= 0:
                raise ValueError("stress.synthetic_clusters.dim must be > 0")
        heavy_cfg = getattr(stress_cfg, "heavy_tail", None)
        if heavy_cfg is not None:
            if not (0.0 <= heavy_cfg.p_tail <= 1.0):
                raise ValueError("stress.heavy_tail.p_tail must be in [0, 1]")
            if heavy_cfg.tail_scale <= 0:
                raise ValueError("stress.heavy_tail.tail_scale must be > 0")


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

    data_dict = dict(data)
    cp_cfg_dict = data_dict.pop("class_packed", None)
    if cp_cfg_dict is None:
        cp_cfg = ClassPackedConfig()
    else:
        if not isinstance(cp_cfg_dict, dict):
            raise ValueError("data.class_packed must be a mapping when provided")
        cp_cfg = ClassPackedConfig(**cp_cfg_dict)
    data_cfg = DataConfig(class_packed=cp_cfg, **data_dict)
    aug_cfg = AugConfig(**aug)
    model_cfg = ModelConfig(**model)
    method_cfg = MethodConfig(**method)
    optim_cfg = OptimConfig(**optim)
    train_cfg = TrainConfig(**train)
    stress_dict = d.get("stress", {})
    if stress_dict is None:
        stress_dict = {}
    if not isinstance(stress_dict, dict):
        raise ValueError("stress section must be a mapping when provided")
    synth_dict = stress_dict.get("synthetic_clusters", {})
    if synth_dict is None:
        synth_dict = {}
    if not isinstance(synth_dict, dict):
        raise ValueError("stress.synthetic_clusters must be a mapping when provided")
    heavy_dict = stress_dict.get("heavy_tail", {})
    if heavy_dict is None:
        heavy_dict = {}
    if not isinstance(heavy_dict, dict):
        raise ValueError("stress.heavy_tail must be a mapping when provided")
    synth_cfg = SyntheticClustersConfig(**synth_dict)
    heavy_cfg = HeavyTailConfig(**heavy_dict)
    stress_cfg = StressConfig(
        synthetic_clusters=synth_cfg,
        heavy_tail=heavy_cfg,
    )

    cfg = Config(
        data=data_cfg,
        aug=aug_cfg,
        model=model_cfg,
        method=method_cfg,
        optim=optim_cfg,
        train=train_cfg,
        stress=stress_cfg,
    )
    validate(cfg)
    return cfg


__all__ = [
    "ClassPackedConfig",
    "DataConfig",
    "AugConfig",
    "ModelConfig",
    "MethodConfig",
    "OptimConfig",
    "TrainConfig",
    "SyntheticClustersConfig",
    "HeavyTailConfig",
    "StressConfig",
    "Config",
    "validate",
    "to_omegaconf",
    "from_omegaconf",
]
