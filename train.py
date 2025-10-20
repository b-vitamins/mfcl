from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

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
from torch.nn.parallel import DistributedDataParallel as DDP

from mfcl.telemetry.timers import StepTimer
from mfcl.utils.dist import (
    get_local_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    unwrap_ddp,
)
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
        bank_device: str | torch.device = "cuda",
    ) -> None:
        self.encoder_getter = encoder_getter
        self.loader = loader
        self.k = int(max(1, k))
        self.temperature = float(max(1e-6, temperature))
        self.every = max(1, int(every_n_epochs))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(bank_device, str):
            bank_device = torch.device(bank_device)
        if bank_device.type == "cuda" and not torch.cuda.is_available():
            bank_device = torch.device("cpu")
        self._bank_device = bank_device

    @torch.no_grad()
    def on_eval_end(self, metrics: dict) -> None:  # pragma: no cover - integration tested
        if not is_main_process():
            return
        epoch = int(metrics.get("epoch", 0) or 0)
        if epoch and (epoch % self.every) != 0:
            return
        encoder = self.encoder_getter().to(self._device)
        encoder.eval()
        bank_feats: list[torch.Tensor] = []
        bank_labels: list[torch.Tensor] = []
        for images, targets in self.loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = encoder(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            bank_feats.append(feats.to(self._bank_device, non_blocking=True))
            bank_labels.append(targets.to(self._bank_device, non_blocking=True))
        if not bank_feats:
            return
        bank_feats_t = torch.cat(bank_feats, dim=0)
        bank_labels_t = torch.cat(bank_labels, dim=0)
        total = bank_feats_t.shape[0]
        top1_sum, count = 0.0, 0
        start = 0
        for images, targets in self.loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = encoder(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            batch = images.size(0)
            end = min(total, start + batch)
            before_feats = bank_feats_t[:start]
            after_feats = bank_feats_t[end:]
            before_labels = bank_labels_t[:start]
            after_labels = bank_labels_t[end:]
            if before_feats.numel() == 0 and after_feats.numel() == 0:
                start = end
                continue
            if before_feats.numel() == 0:
                ref_feats = after_feats
                ref_labels = after_labels
            elif after_feats.numel() == 0:
                ref_feats = before_feats
                ref_labels = before_labels
            else:
                ref_feats = torch.cat((before_feats, after_feats), dim=0)
                ref_labels = torch.cat((before_labels, after_labels), dim=0)
            probs = knn_predict(
                feats, ref_feats, ref_labels, k=self.k, temperature=self.temperature
            )
            _, pred = probs.topk(1, dim=1)
            top1_sum += (pred.squeeze(1) == targets).float().sum().item()
            count += images.size(0)
            start = end
        metrics["knn_top1"] = float(100.0 * top1_sum / max(1, count))


def _maybe_get_encoder(method: torch.nn.Module) -> torch.nn.Module:
    base = unwrap_ddp(method)
    if hasattr(base, "encoder_q"):
        return base.encoder_q
    if hasattr(base, "f_q"):
        return base.f_q
    return getattr(base, "encoder", base)


_CLI_FLAGS: dict[str, bool] = {"print_config": False, "ddp_find_unused": False}


def _infer_steps_per_epoch(loader: DataLoader | None) -> int | None:
    try:
        if loader is None:
            return None
        return len(loader)  # type: ignore[arg-type]
    except Exception:
        return None


def _maybe_autofill_byol_schedule_steps(conf: Config, steps_per_epoch: int | None) -> None:
    method = getattr(conf.method, "name", "").lower()
    if method != "byol":
        return
    schedule = getattr(conf.method, "byol_momentum_schedule", "const").lower()
    if schedule != "cosine":
        return
    if conf.method.byol_momentum_schedule_steps is not None:
        return
    if steps_per_epoch is None:
        raise ValueError(
            "method.byol_momentum_schedule_steps must be set when using cosine "
            "momentum scheduling and the train loader has no finite length."
        )
    total_steps = steps_per_epoch * conf.train.epochs
    if total_steps <= 0:
        raise ValueError(
            "Computed method.byol_momentum_schedule_steps must be > 0. Check train.epochs."
        )
    conf.method.byol_momentum_schedule_steps = int(total_steps)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def _hydra_entry(cfg: DictConfig) -> None:
    init_distributed()
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
    if _CLI_FLAGS.get("print_config", False) and is_main_process():
        print(OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(conf.train.seed, deterministic=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(conf.train.cudnn_bench)

    if torch.cuda.is_available():
        device = torch.device("cuda", get_local_rank())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - defensive
            pass

    train_loader, val_loader = build_data(conf)
    if getattr(conf.train, "prefetch_gpu", False) and device.type == "cuda":
        from mfcl.utils.prefetch import PrefetchLoader

        train_loader = PrefetchLoader(
            train_loader,
            device,
            channels_last=getattr(conf.train, "channels_last", False),
            prefetch_depth=getattr(conf.train, "prefetch_depth", 1),
        )

    gpu_augmentor = None
    if (
        getattr(conf.train, "gpu_augment", False)
        and getattr(conf.aug, "backend", "cpu").lower() == "tv2"
    ):
        from mfcl.transforms.gpu import build_gpu_augmentor

        gpu_augmentor = build_gpu_augmentor(
            conf.method.name,
            conf.aug,
            channels_last=getattr(conf.train, "channels_last", False),
        )

    steps_per_epoch = _infer_steps_per_epoch(train_loader)
    _maybe_autofill_byol_schedule_steps(conf, steps_per_epoch)

    distributed = get_world_size() > 1

    method = build_method(conf)
    method = method.to(device)
    if getattr(conf.train, "channels_last", False):
        try:
            method.to(memory_format=torch.channels_last)
        except (TypeError, RuntimeError):
            pass
    if getattr(conf.train, "compile", False):
        if not hasattr(torch, "compile"):
            raise RuntimeError("train.compile requested but torch.compile is unavailable")
        compile_mode = "max-autotune" if device.type == "cuda" else "reduce-overhead"
        method = torch.compile(method, mode=compile_mode)  # type: ignore[operator]
    if distributed:
        find_unused = _CLI_FLAGS.get("ddp_find_unused", False)
        if torch.cuda.is_available():
            method = DDP(
                method,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=find_unused,
            )
        else:
            method = DDP(method, find_unused_parameters=find_unused)

    optimizer = build_optimizer(conf, method)
    scheduler = build_sched(conf, optimizer, steps_per_epoch=steps_per_epoch)

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
                bank_device=knn_cfg.get("bank_device", "cuda"),
            )
        )

    resume = os.environ.get("MFCL_RESUME", None)
    save_dir = conf.train.save_dir
    if resume is None and save_dir and os.path.isdir(save_dir):
        latest = os.path.join(save_dir, "latest.pt")
        if os.path.exists(latest):
            resume = latest

    if provenance_enabled and save_dir and is_main_process():
        from mfcl.utils.provenance import (
            append_event,
            collect_provenance,
            write_stable_manifest_once,
        )

        prov_dir = Path(save_dir) / "provenance"
        snapshot = collect_provenance(plain_cfg)
        write_stable_manifest_once(prov_dir, snapshot)
        event: Dict[str, Any] = {
            "program": "train",
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "type": "resume" if resume else "start",
        }
        if resume:
            event["resumed_from"] = str(resume)
        append_event(prov_dir, event)

    timing_cfg: dict[str, Any] = {}
    runtime_cfg: dict[str, Any] = {}
    if isinstance(runtime_dict, dict):
        runtime_cfg = runtime_dict
        timing_node = runtime_dict.get("timing", {})
        if isinstance(timing_node, dict):
            timing_cfg = timing_node
    timing_enabled = bool(timing_cfg.get("enabled", True))
    warmup_steps = int(max(0, int(timing_cfg.get("warmup_steps", 50))))
    sample_rate_raw = int(timing_cfg.get("sample_rate", 1))
    sample_rate = 1 if sample_rate_raw <= 0 else sample_rate_raw
    nvtx_enabled = bool(runtime_cfg.get("nvtx", False))

    step_timer: StepTimer | None = None
    if timing_enabled:
        log_path: Path | None = None
        if save_dir:
            log_path = Path(save_dir) / "timings.csv"
        step_timer = StepTimer(
            enabled=timing_enabled,
            warmup_steps=warmup_steps,
            sample_rate=sample_rate,
            log_path=log_path,
            nvtx_enabled=nvtx_enabled,
            is_main=is_main_process(),
        )

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
        channels_last_inputs=getattr(conf.train, "channels_last", False),
        gpu_augmentor=gpu_augmentor,
        timer=step_timer,
    )

    try:
        trainer.fit(
            train_loader,
            val_loader=None,
            epochs=conf.train.epochs,
            resume_path=resume,
            eval_every=1,
            save_every=1,
        )
    finally:
        if step_timer is not None:
            step_timer.close()

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
    if args.prefetch_factor is not None:
        overrides.append(f"data.prefetch_factor={int(args.prefetch_factor)}")
    if args.mp_context:
        overrides.append(f"data.multiprocessing_context={args.mp_context}")
    if args.worker_threads is not None:
        overrides.append(f"data.worker_threads={int(args.worker_threads)}")
    if args.pin_memory is not None:
        overrides.append(f"data.pin_memory={'true' if args.pin_memory else 'false'}")
    if args.pin_memory_device:
        overrides.append(f"data.pin_memory_device={args.pin_memory_device}")
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
    if args.moco_queue_device:
        overrides.append(f"method.moco_queue_device={args.moco_queue_device}")
    if args.moco_queue_dtype:
        overrides.append(f"method.moco_queue_dtype={args.moco_queue_dtype}")
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
    if args.aug_backend:
        overrides.append(f"aug.backend={args.aug_backend}")
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
    if args.prefetch_gpu is not None:
        overrides.append(f"train.prefetch_gpu={'true' if args.prefetch_gpu else 'false'}")
    if args.gpu_augment is not None:
        overrides.append(f"train.gpu_augment={'true' if args.gpu_augment else 'false'}")
    if args.prefetch_depth is not None:
        overrides.append(f"train.prefetch_depth={int(args.prefetch_depth)}")
    if args.channels_last is not None:
        overrides.append(f"train.channels_last={'true' if args.channels_last else 'false'}")
    if args.compile is not None:
        overrides.append(f"train.compile={'true' if args.compile else 'false'}")
    if args.loss_fp32 is not None:
        overrides.append(f"train.loss_fp32={'true' if args.loss_fp32 else 'false'}")
    if args.cudnn_bench is not None:
        overrides.append(f"train.cudnn_bench={'true' if args.cudnn_bench else 'false'}")
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
    if args.knn_bank_device:
        overrides.append(f"knn.bank_device={args.knn_bank_device}")
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
    parser.add_argument("--worker-threads", type=int, help="Torch threads per DataLoader worker.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        help="Number of samples prefetched by each DataLoader worker.",
    )
    parser.add_argument(
        "--mp-context",
        help="torch.multiprocessing start method for DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory-device",
        help="Device passed to DataLoader(pin_memory_device).",
    )
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
    parser.add_argument(
        "--aug-backend",
        choices=["cpu", "tv2"],
        help="Augmentation backend ('cpu' or 'tv2').",
    )
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
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        help="Number of batches to stage on device with GPU prefetch.",
    )
    prefetch_group = parser.add_mutually_exclusive_group()
    prefetch_group.add_argument(
        "--prefetch-gpu",
        dest="prefetch_gpu",
        action="store_true",
        help="Enable asynchronous GPU prefetch of training batches.",
    )
    prefetch_group.add_argument(
        "--no-prefetch-gpu",
        dest="prefetch_gpu",
        action="store_false",
        help="Disable GPU prefetch when enabled in config.",
    )
    parser.set_defaults(prefetch_gpu=None)
    gpu_aug_group = parser.add_mutually_exclusive_group()
    gpu_aug_group.add_argument(
        "--gpu-augment",
        dest="gpu_augment",
        action="store_true",
        help="Enable torchvision v2 GPU augmentation when available.",
    )
    gpu_aug_group.add_argument(
        "--no-gpu-augment",
        dest="gpu_augment",
        action="store_false",
        help="Disable GPU augmentation even if configured.",
    )
    parser.set_defaults(gpu_augment=None)
    channels_group = parser.add_mutually_exclusive_group()
    channels_group.add_argument(
        "--channels-last",
        dest="channels_last",
        action="store_true",
        help="Enable channels-last memory format for model and inputs.",
    )
    channels_group.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="Disable channels-last memory format.",
    )
    parser.set_defaults(channels_last=None)
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Compile the model with torch.compile for optimized kernels.",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile if enabled in the config.",
    )
    parser.set_defaults(compile=None)
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument(
        "--loss-fp32",
        dest="loss_fp32",
        action="store_true",
        help="Force losses to run in float32 (default).",
    )
    loss_group.add_argument(
        "--no-loss-fp32",
        dest="loss_fp32",
        action="store_false",
        help="Allow losses to follow autocast dtype without fp32 promotion.",
    )
    parser.set_defaults(loss_fp32=None)
    bench_group = parser.add_mutually_exclusive_group()
    bench_group.add_argument(
        "--cudnn-bench",
        dest="cudnn_bench",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark.",
    )
    bench_group.add_argument(
        "--no-cudnn-bench",
        dest="cudnn_bench",
        action="store_false",
        help="Disable cudnn benchmark heuristic selection.",
    )
    parser.set_defaults(cudnn_bench=None)
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--run-dir", help="Explicit checkpoint directory (train.save_dir).")
    parser.add_argument(
        "--scheduler-step",
        choices=["batch", "epoch"],
        help="When to step the scheduler (batch or epoch).",
    )
    parser.add_argument("--temperature", type=float, help="InfoNCE / MoCo / SwAV temperature.")
    parser.add_argument("--moco-queue", type=int, help="MoCo queue size.")
    parser.add_argument(
        "--moco-queue-device",
        choices=["cpu", "cuda"],
        help="Device used to store MoCo negatives.",
    )
    parser.add_argument(
        "--moco-queue-dtype",
        choices=["fp32", "fp16"],
        help="Precision used for the MoCo negative queue.",
    )
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
    parser.add_argument(
        "--knn-bank-device",
        choices=["cpu", "cuda"],
        help="Device used to store the kNN feature bank.",
    )
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
        "--print-config",
        action="store_true",
        help="Print the resolved Hydra config before training.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused",
        action="store_true",
        help="Enable DistributedDataParallel(find_unused_parameters=True) for dynamic graphs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    extra = list(getattr(args, "overrides", []))
    overrides = _cli_overrides(args, extra)

    if args.resume:
        os.environ["MFCL_RESUME"] = args.resume
    _CLI_FLAGS["print_config"] = bool(args.print_config)
    _CLI_FLAGS["ddp_find_unused"] = bool(getattr(args, "ddp_find_unused", False))

    argv_backup = sys.argv
    try:
        sys.argv = [argv_backup[0]] + list(overrides)
        _hydra_entry()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
