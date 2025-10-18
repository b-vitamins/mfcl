from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from mfcl.core.config import Config, from_omegaconf, validate
from mfcl.core.factory import build_data, build_method, build_optimizer, build_sched
from mfcl.engines.hooks import Hook, HookList
from mfcl.engines.trainer import Trainer
from mfcl.metrics.knn import knn_predict
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.utils.dist import init_distributed, is_main_process
from mfcl.utils.seed import set_seed


class KNNHook(Hook):
    """Periodic kNN sanity check on a held-out loader."""

    def __init__(
        self,
        encoder_getter: Callable[[], torch.nn.Module],
        loader: DataLoader,
        *,
        k: int,
        temperature: float,
        every_n_epochs: int,
    ) -> None:
        self.encoder_getter = encoder_getter
        self.loader = loader
        self.k = int(max(1, k))
        self.temperature = float(max(1e-6, temperature))
        self.every = max(1, int(every_n_epochs))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def on_eval_end(self, metrics: dict) -> None:  # pragma: no cover - integration tested
        if not is_main_process():
            return
        epoch = int(metrics.get("epoch", 0) or 0)
        if epoch and (epoch % self.every) != 0:
            return
        encoder = self.encoder_getter().to(self._device)
        encoder.eval()
        bank_feats, bank_labels = [], []
        for images, targets in self.loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = encoder(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            bank_feats.append(feats.cpu())
            bank_labels.append(targets.cpu())
        if not bank_feats:
            return
        bank_feats_t = torch.cat(bank_feats, dim=0)
        bank_labels_t = torch.cat(bank_labels, dim=0)
        top1_sum, count = 0.0, 0
        for images, targets in self.loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = encoder(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            probs = knn_predict(
                feats, bank_feats_t, bank_labels_t, k=self.k, temperature=self.temperature
            )
            _, pred = probs.topk(1, dim=1)
            top1_sum += (pred.squeeze(1) == targets).float().sum().item()
            count += images.size(0)
        metrics["knn_top1"] = float(100.0 * top1_sum / max(1, count))


def _maybe_get_encoder(method: torch.nn.Module) -> torch.nn.Module:
    if hasattr(method, "encoder_q"):
        return method.encoder_q
    if hasattr(method, "f_q"):
        return method.f_q
    return getattr(method, "encoder", method)


_CLI_FLAGS: dict[str, bool] = {"print_config": False}


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def _hydra_entry(cfg: DictConfig) -> None:
    if torch.cuda.is_available() and os.environ.get("MFCL_CUDNN_BENCH", "0") == "1":
        torch.backends.cudnn.benchmark = True

    init_distributed()
    conf: Config = from_omegaconf(cfg)
    validate(conf)
    if _CLI_FLAGS.get("print_config", False) and is_main_process():
        print(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(conf.train.seed, deterministic=True)

    train_loader, val_loader = build_data(conf)

    method = build_method(conf)
    optimizer = build_optimizer(conf, method)
    scheduler = build_sched(conf, optimizer)

    console = ConsoleMonitor()
    hooks = HookList()

    knn_cfg = cfg.get("knn", {})
    if knn_cfg.get("enabled", False) and val_loader is not None:
        hooks.add(
            KNNHook(
                lambda: _maybe_get_encoder(method),
                val_loader,
                k=int(knn_cfg.get("k", 20)),
                temperature=float(knn_cfg.get("temperature", 0.07)),
                every_n_epochs=int(knn_cfg.get("every_n_epochs", 10)),
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume = os.environ.get("MFCL_RESUME", None)
    save_dir = conf.train.save_dir
    if resume is None and save_dir and os.path.isdir(save_dir):
        latest = os.path.join(save_dir, "latest.pt")
        if os.path.exists(latest):
            resume = latest

    trainer = Trainer(
        method,
        optimizer,
        scheduler=scheduler,
        console=console,
        device=device,
        hooks=hooks,
        save_dir=save_dir,
        keep_k=3,
        log_interval=conf.train.log_interval,
        accum_steps=1,
        clip_grad=conf.train.grad_clip if conf.train.grad_clip is not None else None,
        scheduler_step_on=conf.train.scheduler_step_on,
    )

    trainer.fit(
        train_loader,
        val_loader=None,
        epochs=conf.train.epochs,
        resume_path=resume,
        eval_every=1,
        save_every=1,
    )

    if is_main_process():
        print(f"[done] epoch={conf.train.epochs} save_dir={save_dir}")


def _cli_overrides(args: argparse.Namespace, extra: Sequence[str]) -> list[str]:
    overrides: list[str] = []
    if args.data:
        overrides.append(f"data={args.data}")
    if args.dataset_root:
        overrides.append(f"data.root={args.dataset_root}")
    if args.synthetic_classes is not None:
        overrides.append(f"data.synthetic_num_classes={int(args.synthetic_classes)}")
    if args.synthetic_train is not None:
        overrides.append(f"data.synthetic_train_size={int(args.synthetic_train)}")
    if args.synthetic_val is not None:
        overrides.append(f"data.synthetic_val_size={int(args.synthetic_val)}")
    if args.num_workers is not None:
        overrides.append(f"data.num_workers={int(args.num_workers)}")
    if args.pin_memory is not None:
        overrides.append(f"data.pin_memory={'true' if args.pin_memory else 'false'}")
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={int(args.batch_size)}")
    if args.shuffle is not None:
        overrides.append(f"data.shuffle={'true' if args.shuffle else 'false'}")
    if args.method:
        overrides.append(f"method={args.method}")
    if args.temperature is not None:
        overrides.append(f"method.temperature={float(args.temperature)}")
    if args.moco_queue is not None:
        overrides.append(f"method.moco_queue={int(args.moco_queue)}")
    if args.swav_prototypes is not None:
        overrides.append(f"method.swav_prototypes={int(args.swav_prototypes)}")
    if args.vicreg_lambda is not None:
        overrides.append(f"method.vicreg_lambda={float(args.vicreg_lambda)}")
    if args.vicreg_mu is not None:
        overrides.append(f"method.vicreg_mu={float(args.vicreg_mu)}")
    if args.vicreg_nu is not None:
        overrides.append(f"method.vicreg_nu={float(args.vicreg_nu)}")
    if args.model:
        overrides.append(f"model={args.model}")
    if args.encoder_dim is not None:
        overrides.append(f"model.encoder_dim={int(args.encoder_dim)}")
    if args.projector_hidden is not None:
        overrides.append(f"model.projector_hidden={int(args.projector_hidden)}")
    if args.projector_out is not None:
        overrides.append(f"model.projector_out={int(args.projector_out)}")
    if args.predictor_hidden is not None:
        overrides.append(f"model.predictor_hidden={int(args.predictor_hidden)}")
    if args.predictor_out is not None:
        overrides.append(f"model.predictor_out={int(args.predictor_out)}")
    if args.augment is not None:
        overrides.append(f"aug={args.augment}")
    if args.img_size is not None:
        overrides.append(f"aug.img_size={int(args.img_size)}")
    if args.local_crops is not None:
        overrides.append(f"aug.local_crops={int(args.local_crops)}")
    if args.local_size is not None:
        overrides.append(f"aug.local_size={int(args.local_size)}")
    if args.optimizer:
        overrides.append(f"optim={args.optimizer}")
    if args.lr is not None:
        overrides.append(f"optim.lr={float(args.lr)}")
    if args.weight_decay is not None:
        overrides.append(f"optim.weight_decay={float(args.weight_decay)}")
    if args.momentum is not None:
        overrides.append(f"optim.momentum={float(args.momentum)}")
    if args.beta1 is not None:
        overrides.append(f"optim.betas.0={float(args.beta1)}")
    if args.beta2 is not None:
        overrides.append(f"optim.betas.1={float(args.beta2)}")
    if args.epochs is not None:
        overrides.append(f"train.epochs={int(args.epochs)}")
    if args.warmup is not None:
        overrides.append(f"train.warmup_epochs={int(args.warmup)}")
    if args.grad_clip is not None:
        overrides.append(f"train.grad_clip={float(args.grad_clip)}")
    if args.log_interval is not None:
        overrides.append(f"train.log_interval={int(args.log_interval)}")
    if args.seed is not None:
        overrides.append(f"train.seed={int(args.seed)}")
    if args.run_dir:
        overrides.append(f"train.save_dir={args.run_dir}")
    if args.scheduler_step:
        overrides.append(f"train.scheduler_step_on={args.scheduler_step}")
    if args.amp is not None:
        overrides.append(f"train.amp={'true' if args.amp else 'false'}")
    if args.cosine is not None:
        overrides.append(f"train.cosine={'true' if args.cosine else 'false'}")
    if args.knn is not None:
        overrides.append("knn=enabled" if args.knn else "knn=disabled")
    if args.knn_k is not None:
        overrides.append(f"knn.k={int(args.knn_k)}")
    if args.knn_temperature is not None:
        overrides.append(f"knn.temperature={float(args.knn_temperature)}")
    if args.knn_period is not None:
        overrides.append(f"knn.every_n_epochs={int(args.knn_period)}")
    overrides.extend(extra)
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a self-supervised model with Hydra-managed configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides like key=value.")
    parser.add_argument("--method", "-m", help="Method config group to use (simclr, moco, byol, ...)")
    parser.add_argument("--data", "-d", help="Data config group (imagenet or synthetic).")
    parser.add_argument("--dataset-root", help="Override data.root pointing to ImageNet directory.")
    parser.add_argument("--synthetic-classes", type=int, help="Synthetic dataset number of classes.")
    parser.add_argument("--synthetic-train", type=int, help="Synthetic dataset train size.")
    parser.add_argument("--synthetic-val", type=int, help="Synthetic dataset val size.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker processes.")
    pm_group = parser.add_mutually_exclusive_group()
    pm_group.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Enable pin_memory")
    pm_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable pin_memory")
    parser.set_defaults(pin_memory=None)
    shuf_group = parser.add_mutually_exclusive_group()
    shuf_group.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle training data")
    shuf_group.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffling")
    parser.set_defaults(shuffle=None)
    parser.add_argument("--batch-size", type=int, help="Mini-batch size per process.")
    parser.add_argument("--augment", help="Augmentation config group (simclr, swav, ...).")
    parser.add_argument("--img-size", type=int, help="Augmentation image size (pixels).")
    parser.add_argument("--local-crops", type=int, help="Number of local crops (for SwAV).")
    parser.add_argument("--local-size", type=int, help="Size of local crops (for SwAV).")
    parser.add_argument("--model", "-b", help="Model config group (resnet18, resnet34, resnet50, timm:...).")
    parser.add_argument("--encoder-dim", type=int, help="Override encoder output dimension.")
    parser.add_argument("--projector-hidden", type=int, help="Projector hidden dimension.")
    parser.add_argument("--projector-out", type=int, help="Projector output dimension.")
    parser.add_argument("--predictor-hidden", type=int, help="Predictor hidden dimension.")
    parser.add_argument("--predictor-out", type=int, help="Predictor output dimension.")
    parser.add_argument("--optimizer", "-o", help="Optimizer config group (sgd, lars, adamw).")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay.")
    parser.add_argument("--momentum", type=float, help="Momentum (SGD/LARS).")
    parser.add_argument("--beta1", type=float, help="AdamW beta1 override.")
    parser.add_argument("--beta2", type=float, help="AdamW beta2 override.")
    parser.add_argument("--epochs", "-e", type=int, help="Number of training epochs.")
    parser.add_argument("--warmup", type=int, help="Warmup epochs before cosine schedule.")
    parser.add_argument("--grad-clip", type=float, help="Gradient clipping max-norm.")
    parser.add_argument("--log-interval", type=int, help="Batches between log updates.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--run-dir", help="Explicit checkpoint directory (train.save_dir).")
    parser.add_argument(
        "--scheduler-step",
        choices=["batch", "epoch"],
        help="When to step the scheduler (batch or epoch).",
    )
    parser.add_argument("--temperature", type=float, help="InfoNCE / MoCo / SwAV temperature.")
    parser.add_argument("--moco-queue", type=int, help="MoCo queue size.")
    parser.add_argument("--swav-prototypes", type=int, help="Number of SwAV prototypes.")
    parser.add_argument("--vicreg-lambda", type=float, help="VICReg invariance coefficient.")
    parser.add_argument("--vicreg-mu", type=float, help="VICReg variance coefficient.")
    parser.add_argument("--vicreg-nu", type=float, help="VICReg covariance coefficient.")
    parser.add_argument("--knn", dest="knn", action="store_true", help="Enable periodic kNN monitor.")
    parser.add_argument("--no-knn", dest="knn", action="store_false", help="Disable kNN monitor.")
    parser.set_defaults(knn=None)
    parser.add_argument("--knn-k", type=int, help="Number of neighbours for kNN monitor.")
    parser.add_argument("--knn-temperature", type=float, help="Softmax temperature for kNN monitor.")
    parser.add_argument("--knn-period", type=int, help="Epoch period for kNN monitor.")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision training.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision training.")
    parser.set_defaults(amp=None)
    cos_group = parser.add_mutually_exclusive_group()
    cos_group.add_argument("--cosine", dest="cosine", action="store_true", help="Use cosine LR schedule.")
    cos_group.add_argument("--no-cosine", dest="cosine", action="store_false", help="Disable cosine LR schedule.")
    parser.set_defaults(cosine=None)
    parser.add_argument("--resume", help="Resume from checkpoint path (sets MFCL_RESUME).")
    parser.add_argument(
        "--cudnn-bench",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark for performance tuning.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved Hydra config before training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    extra = list(getattr(args, "overrides", []))
    overrides = _cli_overrides(args, extra)

    if args.resume:
        os.environ["MFCL_RESUME"] = args.resume
    if args.cudnn_bench:
        os.environ["MFCL_CUDNN_BENCH"] = "1"
    _CLI_FLAGS["print_config"] = bool(args.print_config)

    argv_backup = sys.argv
    try:
        sys.argv = [argv_backup[0]] + list(overrides)
        _hydra_entry()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
