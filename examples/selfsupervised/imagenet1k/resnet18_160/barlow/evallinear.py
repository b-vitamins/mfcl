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
from torchvision import datasets


def _extract_encoder_state(method_state: dict) -> dict:
    prefixes = ["encoder_q.", "encoder_online.", "f_q.", "encoder."]
    for p in prefixes:
        keys = [k for k in method_state.keys() if k.startswith(p)]
        if keys:
            return {k[len(p) :]: method_state[k] for k in keys}
    return method_state


def _build_eval_loaders(cfg: Config):
    tfm = to_tensor_and_norm()
    tr = os.path.join(cfg.data.root, "train")
    vr = os.path.join(cfg.data.root, "val")
    if not (os.path.isdir(tr) and os.path.isdir(vr)):
        raise FileNotFoundError("Expected ImageNet train/val under data.root")
    tds = datasets.ImageFolder(tr, transform=tfm)
    vds = datasets.ImageFolder(vr, transform=tfm)
    tloader = DataLoader(
        tds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    vloader = DataLoader(
        vds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return tloader, vloader


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
    tl, vl = _build_eval_loaders(conf)
    probe = LinearProbe(
        encoder, feature_dim=conf.model.encoder_dim, num_classes=1000, device=device
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
