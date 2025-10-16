from __future__ import annotations
import os
from typing import Any
import torch
from torch.utils.data import DataLoader

from mfcl.engines.evaluator import LinearProbe
from mfcl.utils.seed import set_seed
from mfcl.utils.checkpoint import load_checkpoint
from mfcl.transforms.common import to_tensor_and_norm
from mfcl.core.factory import build_encoder
from mfcl.core.config import from_omegaconf, Config
from mfcl.utils.dist import is_main_process
from torchvision import datasets, transforms as T


def _extract_encoder_state(method_state: dict) -> dict:
    prefixes = [
        "encoder_q.",
        "encoder_online.",
        "f_q.",
        "encoder.",
        "module.encoder.",
        "module.encoder_q.",
    ]
    for p in prefixes:
        keys = [k for k in method_state.keys() if k.startswith(p)]
        if keys:
            return {k[len(p) :]: method_state[k] for k in keys}
    return method_state


def _build_eval_transform(img_size: int):
    scale = int(round(img_size / 0.875))
    return T.Compose([T.Resize(scale), T.CenterCrop(img_size), to_tensor_and_norm()])


def _build_eval_loaders(cfg: Config):
    img_size = getattr(cfg.aug, "img_size", 224)
    tfm = _build_eval_transform(img_size)
    tr = os.path.join(cfg.data.root, "train")
    vr = os.path.join(cfg.data.root, "val")
    if not (os.path.isdir(tr) and os.path.isdir(vr)):
        raise FileNotFoundError("Expected ImageNet train/val under data.root")
    tds = datasets.ImageFolder(tr, transform=tfm)
    vds = datasets.ImageFolder(vr, transform=tfm)
    num_classes = len(tds.classes)
    tloader = DataLoader(
        tds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
    )
    vloader = DataLoader(
        vds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
    )
    return tloader, vloader, num_classes


def main(cfg: Any) -> None:
    conf: Config = from_omegaconf(cfg)
    set_seed(conf.train.seed, deterministic=True)
    ckpt = os.environ.get("MFCL_CKPT", None)
    if ckpt is None:
        try:
            ckpt = cfg.get("ckpt", None)
        except Exception:
            ckpt = None
    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError("Provide checkpoint via MFCL_CKPT or +ckpt=path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(conf)
    state = load_checkpoint(ckpt, strict=True)
    method_state = state.get("method", None)
    if method_state is None:
        raise RuntimeError("Checkpoint missing 'method' state")
    enc_state = _extract_encoder_state(method_state)
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval().requires_grad_(False).to(device)
    tl, vl, num_classes = _build_eval_loaders(conf)
    probe = LinearProbe(
        encoder,
        feature_dim=conf.model.encoder_dim,
        num_classes=num_classes,
        device=device,
    )
    metrics = probe.fit(tl, vl)
    if is_main_process():
        print(
            f"[linear] top1={metrics['top1'] * 100:.2f} top5={metrics['top5'] * 100:.2f} loss={metrics['loss']:.4f}"
        )


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path="../../../../../configs", config_name="defaults", version_base=None
    )
    def _entry(cfg: DictConfig) -> None:  # type: ignore[name-defined]
        main(cfg)

    _entry()
