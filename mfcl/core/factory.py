"""Factories and registries to build MFCL components from config.

This module provides explicit registries and thin builder functions to
instantiate encoders, heads, losses, methods, transforms, data loaders,
optimizers, and learning-rate schedulers. No global magic; everything is
registered via explicit calls and constructed deterministically.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # static typing: use public base for compatibility
    from torch.optim.lr_scheduler import LRScheduler as SchedulerBase  # type: ignore
else:  # runtime import (type not used by checker here)
    from torch.optim.lr_scheduler import _LRScheduler as SchedulerBase  # type: ignore

from .registry import Registry
from .config import Config
from mfcl.methods.base import BaseMethod
from mfcl.utils.dist import get_rank, get_world_size
from mfcl.data.sampler import DistSamplerWrapper


# Public registries. Population happens explicitly near imports in this module
# or by external code importing and adding entries, but never implicitly.
ENCODER_REGISTRY = Registry("encoder")
HEAD_REGISTRY = Registry("head")
METHOD_REGISTRY = Registry("method")
LOSS_REGISTRY = Registry("loss")
TRANSFORM_REGISTRY = Registry("transform")

# -----------------------------------------------------------------------------
# Explicit registrations: encoders, heads, losses, transforms, methods
# -----------------------------------------------------------------------------
try:
    from mfcl.models.encoders.resnet import (
        make_resnet18,
        make_resnet34,
        make_resnet50,
    )

    ENCODER_REGISTRY.add("resnet18", make_resnet18)
    ENCODER_REGISTRY.add("resnet34", make_resnet34)
    ENCODER_REGISTRY.add("resnet50", make_resnet50)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional ResNet encoder dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering ResNet encoders") from err

try:
    from mfcl.models.encoders.timmwrap import TimmEncoder

    ENCODER_REGISTRY.add("timm", TimmEncoder)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional timm encoder dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering timm encoder") from err

try:
    from mfcl.models.heads.projector import Projector
    from mfcl.models.heads.predictor import Predictor

    HEAD_REGISTRY.add("projector", Projector)
    HEAD_REGISTRY.add("predictor", Predictor)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional head dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering heads") from err

try:
    from mfcl.losses.ntxent import NTXentLoss
    from mfcl.losses.mococontrast import MoCoContrastLoss
    from mfcl.losses.byolloss import BYOLLoss
    from mfcl.losses.simsiamloss import SimSiamLoss
    from mfcl.losses.swavloss import SwAVLoss
    from mfcl.losses.barlowtwins import BarlowTwinsLoss
    from mfcl.losses.vicregloss import VICRegLoss

    LOSS_REGISTRY.add("ntxent", NTXentLoss)
    LOSS_REGISTRY.add("mococontrast", MoCoContrastLoss)
    LOSS_REGISTRY.add("byolloss", BYOLLoss)
    LOSS_REGISTRY.add("simsiamloss", SimSiamLoss)
    LOSS_REGISTRY.add("swavloss", SwAVLoss)
    LOSS_REGISTRY.add("barlowtwins", BarlowTwinsLoss)
    LOSS_REGISTRY.add("vicregloss", VICRegLoss)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional loss dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering losses") from err

try:
    from mfcl.transforms.simclr import build_pair_transforms
    from mfcl.transforms.multicrop import build_multicrop_transforms

    TRANSFORM_REGISTRY.add("simclr", build_pair_transforms)
    TRANSFORM_REGISTRY.add("multicrop", build_multicrop_transforms)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional transform dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering transforms") from err

try:
    from mfcl.methods.simclr import SimCLR
    from mfcl.methods.moco import MoCo
    from mfcl.methods.byol import BYOL
    from mfcl.methods.simsiam import SimSiam
    from mfcl.methods.barlow import BarlowTwins
    from mfcl.methods.vicreg import VICReg
    from mfcl.methods.swav import SwAV

    METHOD_REGISTRY.add("simclr", SimCLR)
    METHOD_REGISTRY.add("moco", MoCo)
    METHOD_REGISTRY.add("byol", BYOL)
    METHOD_REGISTRY.add("simsiam", SimSiam)
    METHOD_REGISTRY.add("barlow", BarlowTwins)
    METHOD_REGISTRY.add("vicreg", VICReg)
    METHOD_REGISTRY.add("swav", SwAV)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(
        f"Optional method dependencies unavailable: {err}",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as err:  # pragma: no cover - exercised via dedicated unit test
    raise RuntimeError("Unexpected error while registering methods") from err


def _check_encoder_output_dim(
    encoder: nn.Module, expected_dim: int, img_size: int
) -> None:
    """Sanity-check encoder forward output shape against config.

    Runs a forward pass on a dummy input to ensure the encoder returns a 2-D
    tensor with the expected feature dimension. Assumes square ``img_size`` and
    performs the check on CPU tensors; it does not validate dtype or
    normalization semantics.

    Args:
        encoder: Backbone module.
        expected_dim: Expected feature dimension (D).
        img_size: Square input image size (H=W).

    Raises:
        ValueError: If the encoder output shape is incompatible with the config.
    """
    encoder_was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(2, 3, img_size, img_size)
            out = encoder(x)
        if (
            not isinstance(out, torch.Tensor)
            or out.ndim != 2
            or out.shape[1] != expected_dim
        ):
            raise ValueError(
                "Encoder forward must return [B, D] tensor matching "
                f"cfg.model.encoder_dim. Got shape {tuple(out.shape)} vs expected D={expected_dim}."
            )
    finally:
        encoder.train(encoder_was_training)


def build_encoder(cfg: Config) -> nn.Module:
    """Instantiate encoder backbone from ``cfg.model.encoder``.

    Args:
        cfg: Global configuration with model and aug sections populated.

    Returns:
        nn.Module whose forward returns ``[B, cfg.model.encoder_dim]`` tensor.

    Raises:
        KeyError: If the encoder key is not registered.
        ValueError: If the module's output shape does not match the expected dim.
    """
    enc_key = cfg.model.encoder
    model_name = None
    if isinstance(enc_key, str) and enc_key.startswith("timm:"):
        model_name = enc_key.split(":", 1)[1]
        enc_key = "timm"
    enc_ctor = ENCODER_REGISTRY.get(enc_key)
    # Try to pass norm_feat and model_name when supported; fall back to default ctor.
    try:
        if enc_key == "timm":
            encoder = enc_ctor(
                model_name=model_name,
                pretrained=False,
                norm_feat=getattr(cfg.model, "norm_feat", True),
            )  # type: ignore[misc]
        else:
            encoder = enc_ctor(norm_feat=getattr(cfg.model, "norm_feat", True))  # type: ignore[misc]
    except TypeError:
        encoder = enc_ctor()  # type: ignore[misc]
    _check_encoder_output_dim(encoder, cfg.model.encoder_dim, cfg.aug.img_size)
    return encoder


def build_heads(cfg: Config) -> Dict[str, nn.Module]:
    """Build projector and, if required, predictor heads.

    Args:
        cfg: Global configuration with model and method sections.

    Returns:
        Dict with entries ``{"projector": nn.Module}`` and, when applicable,
        ``{"predictor": nn.Module}``.

    Raises:
        KeyError: If a requested head is not registered.
    """
    heads: Dict[str, nn.Module] = {}
    # Projector is always needed for SSL methods in this repo.
    proj_ctor = HEAD_REGISTRY.get("projector")
    heads["projector"] = proj_ctor(
        cfg.model.encoder_dim,
        cfg.model.projector_hidden,
        cfg.model.projector_out,
        use_bn=True,
    )

    method_name = cfg.method.name
    method = method_name.lower()

    needs_predictor = method in {"byol", "simsiam"}
    # Allow methods to override by requesting predictor explicitly via constructor.
    if needs_predictor:
        pred_ctor = HEAD_REGISTRY.get("predictor")
        heads["predictor"] = pred_ctor(
            cfg.model.projector_out,
            cfg.model.predictor_hidden,
            cfg.model.predictor_out,
            use_bn=True,
        )
    return heads


def build_loss(cfg: Config) -> nn.Module:
    """Instantiate the loss module for the selected method.

    Contract: loss forward must return ``(loss: Tensor, stats: dict[str, Tensor])``.

    Args:
        cfg: Global configuration with method section populated.

    Returns:
        An ``nn.Module`` implementing the loss.
    """
    method_name = cfg.method.name
    method = method_name.lower()
    key_map = {
        "simclr": "ntxent",
        "moco": "mococontrast",
        "byol": "byolloss",
        "simsiam": "simsiamloss",
        "swav": "swavloss",
        "barlow": "barlowtwins",
        "vicreg": "vicregloss",
    }
    loss_key = key_map.get(method)
    if loss_key is None:
        raise KeyError(f"No loss mapping for method '{method_name}'.")
    loss_ctor = LOSS_REGISTRY.get(loss_key)
    # Instantiate with method-relevant arguments when applicable.
    if loss_key == "ntxent":
        loss = loss_ctor(
            temperature=cfg.method.temperature,
            normalize=True,
            mode=cfg.method.ntxent_mode,
        )
    elif loss_key == "mococontrast":
        loss = loss_ctor(temperature=cfg.method.temperature)
    elif loss_key == "swavloss":
        loss = loss_ctor(
            epsilon=0.05,
            sinkhorn_iters=cfg.method.swav_sinkhorn_iters,
            temperature=cfg.method.temperature,
            use_float32_for_sinkhorn=cfg.method.swav_use_fp32_sinkhorn,
            sinkhorn_tol=cfg.method.swav_sinkhorn_tol,
            sinkhorn_max_iters=cfg.method.swav_sinkhorn_max_iters,
        )
    elif loss_key == "barlowtwins":
        loss = loss_ctor(lambda_offdiag=cfg.method.barlow_lambda)
    elif loss_key == "vicregloss":
        loss = loss_ctor(
            lambda_invar=cfg.method.vicreg_lambda,
            mu_var=cfg.method.vicreg_mu,
            nu_cov=cfg.method.vicreg_nu,
        )
    else:
        # byolloss, simsiamloss may not require extra args here.
        loss = loss_ctor()

    # Validate contract by running a tiny forward appropriate for the method.
    fwd = getattr(loss, "forward", None)
    if not callable(fwd):
        raise TypeError(f"Loss class '{loss.__class__.__name__}' has no forward().")

    try:
        with torch.no_grad():
            B, D = 4, int(getattr(cfg.model, "projector_out", 128))
            name = cfg.method.name.lower()
            device = torch.device("cpu")

            if name in {"simclr", "barlow", "vicreg"}:
                zeros = torch.zeros(B, D, device=device)
                out = loss(zeros, zeros)

            elif name in {"byol", "simsiam"}:
                zeros = torch.zeros(B, D, device=device)
                # Prefer 2-arg signature; fallback to 4-arg if needed.
                try:
                    out = loss(zeros, zeros)  # common signature
                except TypeError:
                    # Attempt 4-arg variant used by some SimSiam/BYOL implementations.
                    try:
                        out = loss(zeros, zeros, zeros, zeros)
                    except Exception as te2:
                        raise TypeError(
                            f"Loss '{loss.__class__.__name__}' probe failed for both 2-arg and 4-arg signatures: {te2}"
                        ) from te2

            elif name == "moco":

                class _FakeQueue:
                    def __init__(self, k: int, d: int, device: torch.device):
                        self._buf = torch.zeros(k, d, device=device)

                    def get(self) -> torch.Tensor:
                        return self._buf

                K = max(1024, int(getattr(cfg.method, "moco_queue", 65536)))
                fake_q = _FakeQueue(K, D, device)
                out = loss(
                    torch.zeros(B, D, device=device),
                    torch.zeros(B, D, device=device),
                    fake_q,
                )

            elif name == "swav":
                K = int(getattr(cfg.method, "swav_prototypes", 64))
                logits = [
                    torch.zeros(B, K, device=device),
                    torch.zeros(B, K, device=device),
                ]
                out = loss(logits, (0, 1))

            else:
                out = None
            if out is not None:
                if not (isinstance(out, tuple) and len(out) == 2):
                    raise TypeError(
                        f"Loss '{loss.__class__.__name__}' forward must return (loss, stats)"
                    )
    except Exception as e:
        # Re-raise as TypeError to signal contract violation explicitly.
        raise TypeError(
            f"Loss '{loss.__class__.__name__}' failed contract check: {e}"
        ) from e

    return loss


## No introspection helpers; explicit builders only.


def build_method(cfg: Config) -> BaseMethod:
    """Compose and instantiate a method with encoder and heads.

    Note:
        This factory wires components explicitly instead of pulling classes from
        :data:`METHOD_REGISTRY`. The registry remains public for extension code
        that prefers dynamic composition, but keeping the construction explicit
        here avoids ambiguity around heterogeneous method constructors.
    """
    name = cfg.method.name.lower()
    if name == "simclr":
        encoder = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        projector = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.methods.simclr import SimCLR

        return SimCLR(
            encoder=encoder,
            projector=projector,
            temperature=cfg.method.temperature,
            normalize=True,
            ntxent_mode=cfg.method.ntxent_mode,
            cross_rank_negatives=cfg.method.cross_rank_negatives,
        )

    if name == "moco":
        enc_q = build_encoder(cfg)
        enc_k = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        proj_q = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        proj_k = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        if cfg.method.use_syncbn:
            convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
            enc_q = convert(enc_q)
            enc_k = convert(enc_k)
            proj_q = convert(proj_q)
            proj_k = convert(proj_k)
            if get_world_size() <= 1:
                warnings.warn(
                    "method.use_syncbn requested but world_size == 1; conversion will behave as regular BatchNorm.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        from mfcl.methods.moco import MoCo

        return MoCo(
            encoder_q=enc_q,
            encoder_k=enc_k,
            projector_q=proj_q,
            projector_k=proj_k,
            temperature=cfg.method.temperature,
            momentum=cfg.method.moco_momentum,
            queue_size=cfg.method.moco_queue,
            normalize=True,
            cross_rank_queue=cfg.method.cross_rank_queue,
        )

    if name == "byol":
        f_q = build_encoder(cfg)
        f_k = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        g_q = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        g_k = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.models.heads.predictor import Predictor

        q = Predictor(
            cfg.model.projector_out,
            cfg.model.predictor_hidden,
            cfg.model.predictor_out,
            use_bn=True,
        )
        from mfcl.methods.byol import BYOL

        return BYOL(
            encoder_online=f_q,
            encoder_target=f_k,
            projector_online=g_q,
            projector_target=g_k,
            predictor=q,
            tau_base=cfg.method.byol_tau_base,
            tau_final=cfg.method.byol_tau_final,
            momentum_schedule=cfg.method.byol_momentum_schedule,
            momentum_schedule_steps=cfg.method.byol_momentum_schedule_steps,
            normalize=True,
            variant="cosine",
        )

    if name == "simsiam":
        encoder = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        projector = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.models.heads.predictor import Predictor

        predictor = Predictor(
            cfg.model.projector_out,
            cfg.model.predictor_hidden,
            cfg.model.predictor_out,
            use_bn=True,
        )
        from mfcl.methods.simsiam import SimSiam

        return SimSiam(
            encoder=encoder, projector=projector, predictor=predictor, normalize=True
        )

    if name == "barlow":
        encoder = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        projector = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.methods.barlow import BarlowTwins

        return BarlowTwins(
            encoder=encoder,
            projector=projector,
            lambda_offdiag=cfg.method.barlow_lambda,
        )

    if name == "vicreg":
        encoder = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        projector = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.methods.vicreg import VICReg

        return VICReg(
            encoder=encoder,
            projector=projector,
            lambda_inv=cfg.method.vicreg_lambda,
            mu_var=cfg.method.vicreg_mu,
            nu_cov=cfg.method.vicreg_nu,
        )

    if name == "swav":
        encoder = build_encoder(cfg)
        proj_ctor = HEAD_REGISTRY.get("projector")
        projector = proj_ctor(
            cfg.model.encoder_dim,
            cfg.model.projector_hidden,
            cfg.model.projector_out,
            use_bn=True,
        )
        from mfcl.models.prototypes.swavproto import SwAVPrototypes

        prototypes = SwAVPrototypes(
            num_prototypes=cfg.method.swav_prototypes,
            feat_dim=cfg.model.projector_out,
            normalize=True,
            temperature=1.0,
        )
        from mfcl.methods.swav import SwAV

        return SwAV(
            encoder=encoder,
            projector=projector,
            prototypes=prototypes,
            temperature=cfg.method.temperature,
            epsilon=0.05,
            sinkhorn_iters=cfg.method.swav_sinkhorn_iters,
            normalize_input=True,
            use_float32_for_sinkhorn=cfg.method.swav_use_fp32_sinkhorn,
            sinkhorn_tol=cfg.method.swav_sinkhorn_tol,
            sinkhorn_max_iters=cfg.method.swav_sinkhorn_max_iters,
            codes_queue_size=cfg.method.swav_codes_queue_size,
        )

    raise KeyError(f"Unknown method: {cfg.method.name}")


def build_transforms(cfg: Config) -> Callable[[Any], Dict[str, Any]]:
    """Return an image transform that yields multiple views.

    For two-view methods, returns the SimCLR pair transform. For SwAV, returns
    a multi-crop transform.

    Args:
        cfg: Global configuration with aug and method sections.

    Returns:
        A callable ``img -> dict[str, Tensor]`` mapping.
    """
    if cfg.method.name == "swav":
        ctor = TRANSFORM_REGISTRY.get("multicrop")
        return ctor(cfg.aug)
    ctor = TRANSFORM_REGISTRY.get("simclr")
    return ctor(cfg.aug)


def _build_eval_transforms(cfg: Config) -> Callable:
    """Deterministic single-crop eval transform (no jitter/blur)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(cfg.aug.img_size),
            T.CenterCrop(cfg.aug.img_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _wrap_tensor_transform(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Ensure tensor inputs are converted to PIL before applying ``fn``."""

    if fn is None:
        return lambda img: img

    try:
        from torchvision import transforms as T
    except ImportError as exc:  # pragma: no cover - exercised in optional dep tests
        raise RuntimeError(
            "torchvision is required for synthetic dataset support"
        ) from exc

    to_pil = T.ToPILImage()

    def apply(img: Any) -> Any:
        if isinstance(img, torch.Tensor):
            img_pil = to_pil(img)
        else:
            img_pil = img
        return fn(img_pil)

    return apply


def _build_synthetic_datasets(
    cfg: Config, transform: Callable[[Any], Any], eval_transform: Callable[[Any], Any]
) -> Tuple[Any, Any]:
    try:
        from torchvision.datasets import FakeData
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "torchvision is required for synthetic dataset support"
        ) from exc

    wrapped_train = _wrap_tensor_transform(transform)
    wrapped_eval = _wrap_tensor_transform(eval_transform)

    train_ds = FakeData(
        size=int(cfg.data.synthetic_train_size),
        image_size=(3, cfg.aug.img_size, cfg.aug.img_size),
        num_classes=int(cfg.data.synthetic_num_classes),
        transform=wrapped_train,
    )
    val_ds = FakeData(
        size=int(cfg.data.synthetic_val_size),
        image_size=(3, cfg.aug.img_size, cfg.aug.img_size),
        num_classes=int(cfg.data.synthetic_num_classes),
        transform=wrapped_eval,
    )
    return train_ds, val_ds


def build_data(
    cfg: Config,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and optional validation data loaders.

    Uses torchvision's ``ImageFolder`` when file lists are not provided and can
    fall back to synthetic data via ``torchvision.datasets.FakeData``.
    Collates batches into dicts of multiple views according to the selected
    method.

    Args:
        cfg: Global configuration with data/method/aug populated.

    Returns:
        Tuple ``(train_loader, val_loader_or_none)``.
    """
    # Build transforms via registry
    transform = build_transforms(cfg)
    eval_transform = _build_eval_transforms(cfg)

    dataset_kind = getattr(cfg.data, "name", "imagenet").lower()
    if dataset_kind == "synthetic":
        train_ds, val_ds = _build_synthetic_datasets(cfg, transform, eval_transform)
    else:
        from mfcl.data.imagenet1k import build_imagenet_datasets

        train_ds, val_ds = build_imagenet_datasets(
            root=cfg.data.root,
            train_list=cfg.data.train_list,
            val_list=cfg.data.val_list,
            train_transform=transform,
            val_transform=eval_transform,
        )

    # Collate selection depends on method
    if cfg.method.name == "swav":
        from mfcl.data.collate import collate_multicrop as collate_fn
    else:
        from mfcl.data.collate import collate_pair as collate_fn

    persist_workers = cfg.data.persistent_workers and cfg.data.num_workers > 0
    # PyTorch forbids persistent_workers=True when num_workers==0.

    world_size = get_world_size()
    rank = get_rank()
    distributed = world_size > 1

    train_sampler = None
    if distributed:
        train_sampler = DistSamplerWrapper(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=cfg.data.shuffle,
            seed=cfg.train.seed,
            drop_last=cfg.data.drop_last,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False if train_sampler is not None else cfg.data.shuffle,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persist_workers,
        drop_last=cfg.data.drop_last,
        collate_fn=collate_fn,
    )

    val_loader: Optional[DataLoader] = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=persist_workers,
            drop_last=False,
            collate_fn=None,  # use default; eval_transform yields single-view tensors
        )

    return train_loader, val_loader


def _split_params_for_wd(
    named_params: Iterable[Tuple[str, nn.Parameter]],
) -> List[Dict[str, Any]]:
    """Split parameters into two groups: with and without weight decay."""
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if (
            p.ndim == 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """Construct an optimizer for the given model.

    Supports ``sgd``, ``lars`` (alias to SGD here; LARS adaptation may be added
    later), and ``adamw``.

    Args:
        cfg: Global configuration with optim section.
        model: Module providing parameters.

    Returns:
        A PyTorch optimizer instance.
    """
    params = _split_params_for_wd(model.named_parameters())
    name = cfg.optim.name.lower()

    if name == "sgd" or name == "lars":
        # LARS handled as SGD for now; a proper LARS adapter can be swapped in later.
        if name == "lars":
            warnings.warn(
                "Optimizer 'lars' is currently mapped to SGD; a proper LARS adapter can be swapped later.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.optim.SGD(
            params,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.name}")


def build_sched(
    cfg: Config, opt: torch.optim.Optimizer, *, steps_per_epoch: int | None = None
) -> Optional[SchedulerBase]:
    """Return a learning-rate scheduler with optional warmup.

    The trainer is responsible for stepping the returned scheduler (per epoch or
    per batch). Warmup scales LR linearly for ``train.warmup_epochs`` epochs.

    Args:
        cfg: Global config.
        opt: Optimizer to schedule.

    Returns:
        A PyTorch LR scheduler or ``None`` for a constant LR.
    """
    warmup_epochs = int(max(0, cfg.train.warmup_epochs))
    main_epochs = max(1, cfg.train.epochs - warmup_epochs)

    # Decide whether to construct an epoch- or step-based schedule
    step_on = getattr(cfg.train, "scheduler_step_on", "epoch")
    use_batch = str(step_on).lower() == "batch"

    if use_batch and (steps_per_epoch is not None) and (steps_per_epoch > 0):
        warmup_iters = warmup_epochs * int(steps_per_epoch)
        main_iters = max(1, main_epochs * int(steps_per_epoch))
        if cfg.train.cosine:
            after: Optional[SchedulerBase] = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=main_iters
            )
        else:
            step_sz = max(1, int(0.67 * main_iters))
            after = torch.optim.lr_scheduler.StepLR(opt, step_size=step_sz, gamma=0.1)
        if warmup_iters > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, after], milestones=[warmup_iters]
            )
        return after

    # Epoch-based schedule (default / fallback when steps_per_epoch unknown)
    if cfg.train.cosine:
        after = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_epochs)
    else:
        step_sz = max(1, int(0.67 * main_epochs))
        after = torch.optim.lr_scheduler.StepLR(opt, step_size=step_sz, gamma=0.1)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, after], milestones=[warmup_epochs]
        )
    return after


__all__ = [
    "ENCODER_REGISTRY",
    "HEAD_REGISTRY",
    "METHOD_REGISTRY",
    "LOSS_REGISTRY",
    "TRANSFORM_REGISTRY",
    "build_encoder",
    "build_heads",
    "build_loss",
    "build_method",
    "build_transforms",
    "build_data",
    "build_optimizer",
    "build_sched",
]
