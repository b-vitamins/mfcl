"""Loss fidelity probe utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from mfcl.losses.base import SelfSupervisedLoss
from mfcl.utils.dist import is_main_process, unwrap_ddp


def _flatten_grads(outputs: Any) -> torch.Tensor:
    grads: List[torch.Tensor] = []

    def _collect(obj: Any) -> None:
        if torch.is_tensor(obj) and obj.requires_grad:
            grad = obj.grad.detach() if obj.grad is not None else torch.zeros_like(obj)
            grads.append(grad.reshape(-1))
        elif isinstance(obj, dict):
            for value in obj.values():
                _collect(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _collect(value)

    _collect(outputs)
    if not grads:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat([g.to(torch.float32) for g in grads])


def compare_losses(
    loss_a: SelfSupervisedLoss,
    loss_b: SelfSupervisedLoss,
    model: Any,
    batch: Dict[str, Any],
    beta: float,
) -> Dict[str, float]:
    base_model = unwrap_ddp(model)
    training = base_model.training
    params = list(base_model.parameters())
    requires_grad_state = [p.requires_grad for p in params]
    for param in params:
        param.requires_grad_(False)

    try:
        base_model.eval()
        with torch.no_grad():
            encoder_outputs = base_model.forward_views(batch)

        loss_a.zero_grad(set_to_none=True)
        loss_b.zero_grad(set_to_none=True)

        loss_a_value, extras_a = loss_a.compute_loss(
            batch,
            base_model,
            detach_encoder_outputs=True,
            encoder_outputs=encoder_outputs,
        )
        outputs_a = extras_a.pop("_encoder_outputs", None)
        if outputs_a is None:
            outputs_a = encoder_outputs
        if torch.is_tensor(loss_a_value):
            loss_a_value.backward()
        grad_a = _flatten_grads(outputs_a)

        loss_b_value, extras_b = loss_b.compute_loss(
            batch,
            base_model,
            detach_encoder_outputs=True,
            encoder_outputs=encoder_outputs,
        )
        outputs_b = extras_b.pop("_encoder_outputs", None)
        if outputs_b is None:
            outputs_b = encoder_outputs
        if torch.is_tensor(loss_b_value):
            loss_b_value.backward()
        grad_b = _flatten_grads(outputs_b)

        # Reset grads on encoder outputs to avoid leaking references
        def _zero(obj: Any) -> None:
            if torch.is_tensor(obj) and obj.requires_grad and obj.grad is not None:
                obj.grad = None
            elif isinstance(obj, dict):
                for value in obj.values():
                    _zero(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    _zero(value)

        _zero(outputs_a)
        _zero(outputs_b)

        loss_a_scalar = float(loss_a_value.detach().to(torch.float32).item())
        loss_b_scalar = float(loss_b_value.detach().to(torch.float32).item())
        loss_diff = abs(loss_a_scalar - loss_b_scalar)

        norm_a = torch.linalg.norm(grad_a)
        norm_b = torch.linalg.norm(grad_b)
        denom = max(beta, norm_a.item())
        grad_rel_norm = torch.linalg.norm(grad_a - grad_b).item() / denom
        if norm_a.item() < beta or norm_b.item() < beta:
            grad_cos = 1.0
        else:
            grad_cos = float(F.cosine_similarity(grad_a, grad_b, dim=0).item())

        return {
            "loss_a": loss_a_scalar,
            "loss_b": loss_b_scalar,
            "loss_diff": loss_diff,
            "grad_cos": grad_cos,
            "grad_rel_norm": grad_rel_norm,
        }
    finally:
        for param, req in zip(params, requires_grad_state):
            param.requires_grad_(req)
        if training:
            base_model.train()


class FidelityProbe:
    """Periodic fidelity probe that logs loss/gradient deltas."""

    def __init__(
        self,
        loss_a: SelfSupervisedLoss,
        loss_b: SelfSupervisedLoss,
        *,
        interval_steps: int,
        beta: float,
        log_path: Path | None,
    ) -> None:
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.interval = max(1, int(interval_steps))
        self.beta = float(beta)
        self.log_path = log_path
        self._header_written = False

    def maybe_log(self, *, step: int, epoch: int, batch: Dict[str, Any], model: Any) -> None:
        if step <= 0 or (step % self.interval) != 0:
            return
        if not is_main_process():
            return
        metrics = compare_losses(self.loss_a, self.loss_b, model, batch, self.beta)
        metrics.update({"step": int(step), "epoch": int(epoch)})
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = (
            "{step},{epoch},{loss_a:.6f},{loss_b:.6f},{loss_diff:.6f},{grad_cos:.6f},{grad_rel_norm:.6f}\n".format(
                **metrics
            )
        )
        mode = "a"
        with self.log_path.open(mode) as fh:
            if not self._header_written and self.log_path.stat().st_size == 0:
                fh.write(
                    "step,epoch,loss_a,loss_b,loss_diff,grad_cos,grad_rel_norm\n"
                )
                self._header_written = True
            fh.write(line)


__all__ = ["compare_losses", "FidelityProbe"]

