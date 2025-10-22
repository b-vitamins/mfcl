"""Configuration options for :class:`mfcl.engines.trainer.Trainer`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch

from mfcl.engines.hooks import Hook, HookList
from mfcl.runtime.beta_ctrl import BetaController, BetaControllerLogger
from mfcl.runtime.budget import BudgetTracker
from mfcl.telemetry.memory import MemoryMonitor
from mfcl.telemetry.power import PowerMonitor
from mfcl.telemetry.hardness import HardnessMonitor
from mfcl.telemetry.stability import StabilitySentry
from mfcl.telemetry.timers import StepTimer
from mfcl.utils.amp import AmpScaler
from mfcl.utils.consolemonitor import ConsoleMonitor

if torch.__version__ >= "1.13":  # pragma: no cover - torch typing guard
    TorchDevice = torch.device
else:  # pragma: no cover
    TorchDevice = torch.device


@dataclass(frozen=True)
class TrainerOptions:
    """Optional knobs that customise :class:`Trainer` behaviour.

    The dataclass intentionally mirrors the keyword arguments historically accepted
    by :class:`Trainer.__init__`. Instances are immutable so that callers can safely
    share a template and customise per invocation via :func:`dataclasses.replace`.
    """

    console: ConsoleMonitor | None = None
    device: TorchDevice | None = None
    scaler: AmpScaler | None = None
    hooks: Hook | HookList | None = None
    save_dir: str | None = None
    keep_k: int = 3
    log_interval: int = 1
    accum_steps: int = 1
    clip_grad: float | None = None
    scheduler_step_on: str = "batch"
    guard_grad_nan: bool = False
    channels_last_inputs: bool = False
    gpu_augmentor: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None
    timer: StepTimer | None = None
    memory_monitor: MemoryMonitor | None = None
    energy_monitor: PowerMonitor | None = None
    budget_tracker: BudgetTracker | None = None
    fidelity_probe: Any | None = None
    hardness_monitor: HardnessMonitor | None = None
    stability_sentry: StabilitySentry | None = None
    mixture_estimator: Any | None = None
    topr_monitor: Any | None = None
    beta_controller: BetaController | None = None
    beta_controller_logger: BetaControllerLogger | None = None
    third_moment_sketch: Any | None = None
