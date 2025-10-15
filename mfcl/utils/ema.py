"""Exponential moving average updater for momentum/target encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class MomentumUpdater:
    """Exponential moving average parameter updater for target/momentum encoders."""

    def __init__(self, online: nn.Module, target: nn.Module, momentum: float = 0.996):
        """Initialize the updater.

        Args:
            online: Source of parameters.
            target: Destination updated as m*target + (1-m)*online.
            momentum: EMA coefficient in [0,1).
        """
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the range [0, 1)")

        # Validate that modules share identical parameter/buffer structure so that
        # zip-based updates do not silently drop values when one module has more
        # elements than the other (which could happen when architectures diverge).
        online_param_count = sum(1 for _ in online.parameters())
        target_param_count = sum(1 for _ in target.parameters())
        if online_param_count != target_param_count:
            raise ValueError(
                "online and target modules must expose the same number of parameters"
            )
        online_buffer_count = sum(1 for _ in online.buffers())
        target_buffer_count = sum(1 for _ in target.buffers())
        if online_buffer_count != target_buffer_count:
            raise ValueError(
                "online and target modules must expose the same number of buffers"
            )

        self.online = online
        self.target = target
        self.m = float(momentum)

    @torch.no_grad()
    def copy_params(self) -> None:
        """Hard copy online params to target (for initialization)."""
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.copy_(p_o.data)
        # Also copy buffers (e.g., BatchNorm running stats)
        for b_t, b_o in zip(self.target.buffers(), self.online.buffers()):
            b_t.data.copy_(b_o.data)

    @torch.no_grad()
    def update(self) -> None:
        """EMA update for all parameters and buffers with matching order."""
        m = self.m
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))
        for b_t, b_o in zip(self.target.buffers(), self.online.buffers()):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(m).add_(b_o.data, alpha=(1.0 - m))
            else:
                # For integer buffers (e.g., BatchNorm num_batches_tracked), copy
                b_t.data.copy_(b_o.data)


__all__ = ["MomentumUpdater"]
