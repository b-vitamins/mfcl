from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from mfcl.core.config import Config, from_omegaconf, validate
from mfcl.core.factory import build_encoder
from mfcl.engines.evaluator import LinearProbe
from mfcl.transforms.common import to_tensor_and_norm
from mfcl.utils.checkpoint import load_checkpoint
from mfcl.utils.dist import is_main_process
from mfcl.utils.seed import set_seed


def _build_eval_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            to_tensor_and_norm(),
        ]
    )


def _build_fake_transform(img_size: int):
    to_pil = T.ToPILImage()
    base = _build_eval_transform(img_size)

    def apply(img):
        if isinstance(img, torch.Tensor):
            img_pil = to_pil(img)
        else:
            img_pil = img
        return base(img_pil)

    return apply


def _build_imagefolder_loaders(conf: Config, tfm: T.Compose) -> Tuple[DataLoader, DataLoader, int]:
    root = conf.data.root
    train_root = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
    val_root = os.path.join(root, "val")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"ImageNet train directory not found at: {train_root}")
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"ImageNet val directory not found at: {val_root}")
    train_ds = datasets.ImageFolder(train_root, transform=tfm)
    val_ds = datasets.ImageFolder(val_root, transform=tfm)
    num_classes = len(train_ds.classes)
    persistent = bool(conf.data.persistent_workers and conf.data.num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=conf.data.batch_size,
        shuffle=True,
        num_workers=conf.data.num_workers,
        pin_memory=conf.data.pin_memory,
        persistent_workers=persistent,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=conf.data.batch_size,
        shuffle=False,
        num_workers=conf.data.num_workers,
        pin_memory=conf.data.pin_memory,
        persistent_workers=persistent,
        drop_last=False,
    )
    return train_loader, val_loader, num_classes


def _build_synthetic_loaders(conf: Config, img_size: int) -> Tuple[DataLoader, DataLoader, int]:
    from torchvision.datasets import FakeData

    transform = _build_fake_transform(img_size)
    train_ds = FakeData(
        size=int(conf.data.synthetic_train_size),
        image_size=(3, img_size, img_size),
        num_classes=int(conf.data.synthetic_num_classes),
        transform=transform,
    )
    val_ds = FakeData(
        size=int(conf.data.synthetic_val_size),
        image_size=(3, img_size, img_size),
        num_classes=int(conf.data.synthetic_num_classes),
        transform=transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=conf.data.batch_size,
        shuffle=True,
        num_workers=conf.data.num_workers,
        pin_memory=conf.data.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=conf.data.batch_size,
        shuffle=False,
        num_workers=conf.data.num_workers,
        pin_memory=conf.data.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, int(conf.data.synthetic_num_classes)


def _build_eval_loaders(conf: Config) -> Tuple[DataLoader, DataLoader, int]:
    img_size = getattr(conf.aug, "img_size", 224)
    dataset_kind = getattr(conf.data, "name", "imagenet").lower()
    if dataset_kind == "synthetic":
        return _build_synthetic_loaders(conf, img_size)
    tfm = _build_eval_transform(img_size)
    return _build_imagefolder_loaders(conf, tfm)


_CLI_STATE: dict[str, object] = {"print_config": False, "checkpoint": None}


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def _hydra_entry(cfg: DictConfig) -> None:
    conf: Config = from_omegaconf(cfg)
    try:
        plain_cfg_obj = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        plain_cfg: Dict[str, Any]
        if isinstance(plain_cfg_obj, dict):
            plain_cfg = plain_cfg_obj  # type: ignore[assignment]
        else:
            plain_cfg = {}
    except Exception:
        plain_cfg = {}
    runtime_dict = plain_cfg.get("runtime") if isinstance(plain_cfg, dict) else {}
    provenance_enabled = True
    if isinstance(runtime_dict, dict):
        provenance_enabled = bool(runtime_dict.get("provenance", True))
    else:
        try:
            runtime_node = cfg.get("runtime")  # type: ignore[call-arg]
            if runtime_node is not None and hasattr(runtime_node, "get"):
                provenance_enabled = bool(runtime_node.get("provenance", True))
        except Exception:
            provenance_enabled = True
    validate(conf)
    if _CLI_STATE.get("print_config") and is_main_process():
        print(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(conf.train.seed, deterministic=True)

    ckpt = _CLI_STATE.get("checkpoint") or os.environ.get("MFCL_CKPT")
    if not ckpt:
        ckpt = cfg.get("checkpoint") or cfg.get("ckpt")
    if not ckpt:
        raise FileNotFoundError("Provide checkpoint path via --checkpoint, MFCL_CKPT, or +checkpoint=/path/to/ckpt.pt")
    if not os.path.exists(str(ckpt)):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if provenance_enabled and is_main_process():
        from mfcl.utils.provenance import (
            append_event,
            collect_provenance,
            write_stable_manifest_once,
        )

        base_dir = Path(conf.train.save_dir) if conf.train.save_dir else Path(str(ckpt)).resolve().parent
        prov_dir = base_dir / "provenance"
        snapshot = collect_provenance(plain_cfg)
        write_stable_manifest_once(prov_dir, snapshot)
        append_event(
            prov_dir,
            {
                "program": "eval",
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "type": "resume",
                "resumed_from": str(ckpt),
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = build_encoder(conf)
    state = load_checkpoint(str(ckpt), strict=True)
    method_state = state.get("method") if isinstance(state, dict) else None
    if method_state is None:
        raise RuntimeError("Checkpoint missing 'method' state")
    prefixes = ["encoder_q.", "encoder_online.", "f_q.", "encoder.", "module.encoder.", "module.encoder_q."]
    for prefix in prefixes:
        keys = [k for k in method_state.keys() if k.startswith(prefix)]
        if keys:
            filtered = {k[len(prefix):]: method_state[k] for k in keys}
            encoder.load_state_dict(filtered, strict=False)
            break
    else:
        encoder.load_state_dict(method_state, strict=False)
    encoder.eval().requires_grad_(False).to(device)

    train_loader, val_loader, num_classes = _build_eval_loaders(conf)

    linear_cfg = cfg.get("linear", {})
    milestones = tuple(int(m) for m in linear_cfg.get("milestones", [30, 60]))
    batch_override = linear_cfg.get("batch_size")
    probe = LinearProbe(
        encoder,
        feature_dim=conf.model.encoder_dim,
        num_classes=num_classes,
        device=device,
        lr=float(linear_cfg.get("lr", 30.0)),
        epochs=int(linear_cfg.get("epochs", 90)),
        momentum=float(linear_cfg.get("momentum", 0.9)),
        weight_decay=float(linear_cfg.get("weight_decay", 0.0)),
        milestones=milestones,
        batch_size_override=int(batch_override) if batch_override else None,
        feature_norm=bool(linear_cfg.get("feature_norm", False)),
        amp=bool(linear_cfg.get("amp", False)),
        use_scaler=bool(linear_cfg.get("use_scaler", False)),
    )

    metrics = probe.fit(train_loader, val_loader)
    if is_main_process():
        print(
            f"[linear] top1={metrics['top1'] * 100:.2f} top5={metrics['top5'] * 100:.2f} loss={metrics['loss']:.4f}"
        )


def _cli_overrides(args: argparse.Namespace, extra: Sequence[str]) -> list[str]:
    overrides: list[str] = []
    if args.data:
        overrides.append(f"data={args.data}")
    if args.dataset_root:
        overrides.append(f"data.root={args.dataset_root}")
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={int(args.batch_size)}")
    if args.num_workers is not None:
        overrides.append(f"data.num_workers={int(args.num_workers)}")
    if args.pin_memory is not None:
        overrides.append(f"data.pin_memory={'true' if args.pin_memory else 'false'}")
    if args.synthetic_classes is not None:
        overrides.append(f"data.synthetic_num_classes={int(args.synthetic_classes)}")
    if args.synthetic_train is not None:
        overrides.append(f"data.synthetic_train_size={int(args.synthetic_train)}")
    if args.synthetic_val is not None:
        overrides.append(f"data.synthetic_val_size={int(args.synthetic_val)}")
    if args.model:
        overrides.append(f"model={args.model}")
    if args.encoder_dim is not None:
        overrides.append(f"model.encoder_dim={int(args.encoder_dim)}")
    if args.img_size is not None:
        overrides.append(f"aug.img_size={int(args.img_size)}")
    if args.epochs is not None:
        overrides.append(f"linear.epochs={int(args.epochs)}")
    if args.lr is not None:
        overrides.append(f"linear.lr={float(args.lr)}")
    if args.momentum is not None:
        overrides.append(f"linear.momentum={float(args.momentum)}")
    if args.weight_decay is not None:
        overrides.append(f"linear.weight_decay={float(args.weight_decay)}")
    if args.milestones:
        ms = ",".join(str(int(m)) for m in args.milestones)
        overrides.append(f"linear.milestones=[{ms}]")
    if args.linear_batch_size is not None:
        overrides.append(f"linear.batch_size={int(args.linear_batch_size)}")
    if args.feature_norm is not None:
        overrides.append(f"linear.feature_norm={'true' if args.feature_norm else 'false'}")
    if args.amp is not None:
        overrides.append(f"linear.amp={'true' if args.amp else 'false'}")
    if args.use_scaler is not None:
        overrides.append(f"linear.use_scaler={'true' if args.use_scaler else 'false'}")
    overrides.extend(extra)
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen linear evaluation on a pretrained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides like key=value.")
    parser.add_argument("--checkpoint", "-c", help="Path to the pretrained checkpoint.")
    parser.add_argument("--data", "-d", help="Data config group (imagenet or synthetic).")
    parser.add_argument("--dataset-root", help="Override data.root pointing to ImageNet directory.")
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation dataloaders.")
    parser.add_argument("--num-workers", type=int, help="Number of dataloader workers.")
    pm_group = parser.add_mutually_exclusive_group()
    pm_group.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    pm_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--synthetic-classes", type=int, help="Synthetic dataset number of classes.")
    parser.add_argument("--synthetic-train", type=int, help="Synthetic train size.")
    parser.add_argument("--synthetic-val", type=int, help="Synthetic validation size.")
    parser.add_argument("--model", "-b", help="Model config group (resnet18, resnet34, ...).")
    parser.add_argument("--encoder-dim", type=int, help="Encoder dimensionality override.")
    parser.add_argument("--img-size", type=int, help="Center crop size for evaluation.")
    parser.add_argument("--epochs", type=int, help="Linear probe training epochs.")
    parser.add_argument("--lr", type=float, help="Linear probe learning rate.")
    parser.add_argument("--momentum", type=float, help="Linear probe SGD momentum.")
    parser.add_argument("--weight-decay", type=float, help="Linear probe weight decay.")
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="*",
        help="LR milestone epochs (space separated).",
    )
    parser.add_argument(
        "--linear-batch-size",
        type=int,
        help="Override batch size used by the linear probe (linear.batch_size).",
    )
    fn_group = parser.add_mutually_exclusive_group()
    fn_group.add_argument("--feature-norm", dest="feature_norm", action="store_true")
    fn_group.add_argument("--no-feature-norm", dest="feature_norm", action="store_false")
    parser.set_defaults(feature_norm=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    scaler_group = parser.add_mutually_exclusive_group()
    scaler_group.add_argument("--use-scaler", dest="use_scaler", action="store_true")
    scaler_group.add_argument("--no-scaler", dest="use_scaler", action="store_false")
    parser.set_defaults(use_scaler=None)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved Hydra config before evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    extra = list(getattr(args, "overrides", []))
    overrides = _cli_overrides(args, extra)

    _CLI_STATE["print_config"] = bool(args.print_config)
    if args.checkpoint:
        _CLI_STATE["checkpoint"] = args.checkpoint

    argv_backup = sys.argv
    try:
        sys.argv = [argv_backup[0]] + list(overrides)
        _hydra_entry()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
