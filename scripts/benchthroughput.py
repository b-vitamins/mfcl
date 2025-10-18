import argparse
import time
from typing import Tuple

import torch
from omegaconf import OmegaConf

from mfcl.core.factory import build_method
from mfcl.core.config import from_omegaconf, Config
from mfcl.utils.amp import AmpScaler


def _make_cfg(encoder: str, method: str, img: int, batch: int):
    data = {
        "root": "/dev/null",
        "train_list": None,
        "val_list": None,
        "num_workers": 0,
        "batch_size": batch,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
        "drop_last": True,
    }
    aug = {
        "img_size": img,
        "global_crops": 2,
        "local_crops": 4,
        "local_size": 96,
        "jitter_strength": 0.4,
        "blur_prob": 0.5,
        "gray_prob": 0.2,
        "solarize_prob": 0.0,
    }
    model = {
        "encoder": encoder,
        "encoder_dim": 512
        if encoder in ("resnet18", "resnet34") or encoder.startswith("timm:")
        else 2048,
        "projector_hidden": 2048,
        "projector_out": 128,
        "predictor_hidden": 1024,
        "predictor_out": 256,
        "norm_feat": True,
    }
    method_node = {
        "name": method,
        "temperature": 0.1,
        "moco_momentum": 0.999,
        "moco_queue": 4096,
        "swav_prototypes": 300,
        "swav_sinkhorn_iters": 3,
        "barlow_lambda": 5e-3,
        "vicreg_lambda": 25.0,
        "vicreg_mu": 25.0,
        "vicreg_nu": 1.0,
    }
    optim = {
        "name": "sgd",
        "lr": 0.05,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
    }
    train = {
        "epochs": 1,
        "warmup_epochs": 0,
        "cosine": False,
        "amp": True,
        "grad_clip": None,
        "save_dir": "runs/bench",
        "seed": 42,
        "log_interval": 1,
        "scheduler_step_on": "batch",
    }
    cfg = {
        "data": data,
        "aug": aug,
        "model": model,
        "method": method_node,
        "optim": optim,
        "train": train,
    }
    return OmegaConf.create(cfg)


def _make_synthetic_batch(method: str, batch: int, img: int) -> Tuple[dict, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rnd(h: int) -> torch.Tensor:
        return torch.randn(batch, 3, h, h, device=device, dtype=torch.float32)

    if method == "swav":
        crops = [rnd(img) for _ in range(2)] + [rnd(96) for _ in range(4)]
        imgs_per_step = sum(c.size(0) for c in crops)
        return {"crops": crops, "code_crops": (0, 1)}, imgs_per_step
    else:
        batch_dict = {
            "view1": rnd(img),
            "view2": rnd(img),
            "index": torch.arange(batch, device=device),
        }
        return batch_dict, 2 * batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--method", default="simclr")
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    args = ap.parse_args()

    cfg = _make_cfg(args.encoder, args.method, args.img, args.batch)
    tcfg: Config = from_omegaconf(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = build_method(tcfg).to(device)
    method.train()
    opt = torch.optim.SGD(method.parameters(), lr=0.1, momentum=0.9)
    scaler = AmpScaler(enabled=args.amp)

    batch_dict, imgs_per_step = _make_synthetic_batch(args.method, args.batch, args.img)

    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Warmup
    for i in range(args.warmup):
        opt.zero_grad(set_to_none=True)
        with scaler.autocast():
            stats = method.step(batch_dict)
            loss = stats["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        if hasattr(method, "on_optimizer_step"):
            method.on_optimizer_step()

    # Timed
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with scaler.autocast():
            stats = method.step(batch_dict)
            loss = stats["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        if hasattr(method, "on_optimizer_step"):
            method.on_optimizer_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    imgs_s = imgs_per_step * args.steps / dt
    ms_step = 1000.0 * dt / args.steps
    mem_gb = (
        (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    )
    print(
        f"{args.method} {args.encoder} b={args.batch} amp={args.amp} -> {imgs_s:.1f} imgs/s, {ms_step:.2f} ms/step, mem={mem_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
