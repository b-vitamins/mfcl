from __future__ import annotations
import os
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover - import-time only
    from omegaconf import DictConfig
import torch

from mfcl.core.factory import build_data, build_method, build_optimizer, build_sched
from mfcl.core.config import validate, from_omegaconf, Config
from mfcl.utils.seed import set_seed
from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.engines.hooks import Hook, HookList
from mfcl.utils.dist import init_distributed, is_main_process
from torchvision import datasets, transforms as T
from mfcl.transforms.common import to_tensor_and_norm
from torch.utils.data import DataLoader
from mfcl.metrics.knn import knn_predict

if torch.cuda.is_available() and os.environ.get("MFCL_CUDNN_BENCH", "0") == "1":
    torch.backends.cudnn.benchmark = True


def main(cfg: Any) -> None:
    oc: DictConfig = cfg  # for type clarity
    conf: Config = from_omegaconf(oc)
    init_distributed()
    validate(conf)
    set_seed(conf.train.seed, deterministic=True)

    train_loader, _ = build_data(conf)
    method = build_method(conf)
    optimizer = build_optimizer(conf, method)
    scheduler = build_sched(conf, optimizer)
    console = ConsoleMonitor()
    hooks = HookList()

    # Optional kNN sanity-check on rank 0 every few evals
    class KNNHook(Hook):
        def __init__(self, encoder_getter, val_loader, k: int = 20, temperature: float = 0.07, every: int = 10):
            self.encoder_getter = encoder_getter
            self.val_loader = val_loader
            self.k = int(k)
            self.t = float(temperature)
            self.every = int(every)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._calls = 0

        @torch.no_grad()
        def on_epoch_end(self, metrics):
            if not is_main_process():
                return
            self._calls += 1
            if (self._calls % self.every) != 0:
                return
            enc = self.encoder_getter()
            enc.eval().to(self._device)
            bank_feats, bank_labels = [], []
            for images, targets in self.val_loader:
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                feats = enc(images)
                feats = torch.nn.functional.normalize(feats, dim=1)
                bank_feats.append(feats)
                bank_labels.append(targets)
            bank_feats = torch.cat(bank_feats, dim=0)
            bank_labels = torch.cat(bank_labels, dim=0)
            top1_sum, count = 0.0, 0
            for images, targets in self.val_loader:
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                feats = enc(images)
                feats = torch.nn.functional.normalize(feats, dim=1)
                probs = knn_predict(feats, bank_feats, bank_labels, k=self.k, temperature=self.t)
                _, pred = probs.topk(1, dim=1)
                top1_sum += (pred.squeeze(1) == targets).float().sum().item()
                count += images.size(0)
            metrics["knn_top1"] = float(100.0 * top1_sum / max(1, count))

    def _eval_transform(img_size: int) -> T.Compose:
        scale = int(round(img_size / 0.875))
        return T.Compose([T.Resize(scale), T.CenterCrop(img_size), to_tensor_and_norm()])

    def _build_eval_loader(c: Config) -> DataLoader | None:
        val_root = os.path.join(c.data.root, "val")
        if not os.path.isdir(val_root):
            return None
        tfm = _eval_transform(getattr(c.aug, "img_size", 224))
        ds = datasets.ImageFolder(val_root, transform=tfm)
        return DataLoader(
            ds,
            batch_size=c.data.batch_size,
            shuffle=False,
            num_workers=c.data.num_workers,
            pin_memory=c.data.pin_memory,
            persistent_workers=c.data.persistent_workers,
        )

    eval_loader = _build_eval_loader(conf)
    if eval_loader is not None:
        def encoder_getter():
            # swav method wraps encoder as `encoder`
            return getattr(method, "encoder", method)

        hooks.add(KNNHook(encoder_getter, eval_loader, k=20, temperature=0.07, every=10))

    resume = os.environ.get("MFCL_RESUME", None)
    if resume is None and os.path.isdir(conf.train.save_dir):
        cand = os.path.join(conf.train.save_dir, "latest.pt")
        if os.path.exists(cand):
            resume = cand

    trainer = Trainer(
        method,
        optimizer,
        scheduler=scheduler,
        console=console,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hooks=hooks,
        save_dir=conf.train.save_dir,
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
        print("[done] epoch=%d save_dir=%s" % (conf.train.epochs, conf.train.save_dir))


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path="../../../../../configs", config_name="defaults", version_base=None
    )
    def _entry(cfg: DictConfig) -> None:  # type: ignore[name-defined]
        main(cfg)

    _entry()
