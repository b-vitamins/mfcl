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
from mfcl.engines.hooks import HookList
from mfcl.utils.dist import init_distributed, is_main_process


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
