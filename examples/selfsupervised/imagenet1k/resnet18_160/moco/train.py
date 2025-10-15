from __future__ import annotations
import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from mfcl.core.factory import build_data, build_method, build_optimizer, build_sched
from mfcl.core.config import validate, from_omegaconf, Config
from mfcl.utils.seed import set_seed
from mfcl.engines.trainer import Trainer
from mfcl.utils.consolemonitor import ConsoleMonitor
from mfcl.engines.hooks import Hook, HookList
from mfcl.utils.dist import init_distributed, is_main_process

from torchvision import datasets
from mfcl.transforms.common import to_tensor_and_norm
from mfcl.metrics.knn import knn_predict


class KNNHook(Hook):
    def __init__(
        self,
        encoder_getter,
        val_loader,
        k: int = 20,
        temperature: float = 0.07,
        every: int = 10,
    ):
        self.encoder_getter = encoder_getter
        self.val_loader = val_loader
        self.k = int(k)
        self.t = float(temperature)
        self.every = int(every)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def on_eval_end(self, metrics):
        epoch = metrics.get("epoch", 0)
        if (epoch % self.every) != 0:
            return
        enc = self.encoder_getter()
        enc.eval().to(self._device)
        bank_feats, bank_labels = [], []
        for images, targets in self.val_loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = enc(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            bank_feats.append(feats.cpu())
            bank_labels.append(targets.cpu())
        bank_feats = torch.cat(bank_feats, dim=0).to(self._device)
        bank_labels = torch.cat(bank_labels, dim=0).to(self._device)
        top1_sum, count = 0.0, 0
        for images, targets in self.val_loader:
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            feats = enc(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            probs = knn_predict(
                feats, bank_feats, bank_labels, k=self.k, temperature=self.t
            )
            _, pred = probs.topk(1, dim=1)
            top1_sum += (pred.squeeze(1) == targets).float().sum().item()
            count += images.size(0)
        metrics["knn_top1"] = float(100.0 * top1_sum / max(1, count))


def _build_eval_loader(cfg: DictConfig) -> DataLoader | None:
    val_root = os.path.join(cfg.data.root, "val")
    if not os.path.isdir(val_root):
        return None
    tfm = to_tensor_and_norm()
    ds = datasets.ImageFolder(val_root, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        drop_last=False,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b], 0),
            torch.tensor([x[1] for x in b], dtype=torch.long),
        ),
    )
    return loader


@hydra.main(
    config_path="../../../../../configs", config_name="defaults", version_base=None
)
def main(cfg: DictConfig) -> None:
    conf: Config = from_omegaconf(cfg)
    init_distributed()
    validate(conf)
    set_seed(conf.train.seed, deterministic=True)

    train_loader, _ = build_data(conf)
    eval_loader = _build_eval_loader(cfg)

    method = build_method(conf)
    optimizer = build_optimizer(conf, method)
    scheduler = build_sched(conf, optimizer)

    console = ConsoleMonitor()
    hooks = HookList()
    if eval_loader is not None:

        def encoder_getter():
            if hasattr(method, "encoder_q"):
                return method.encoder_q
            if hasattr(method, "f_q"):
                return method.f_q
            return getattr(method, "encoder", method)

        hooks.add(
            KNNHook(encoder_getter, eval_loader, k=20, temperature=0.07, every=10)
        )

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
    main()
