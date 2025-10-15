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
        params_online = list(online.parameters())
        params_target = list(target.parameters())
        if len(params_online) != len(params_target):
            raise ValueError(
                "online and target modules must expose the same number of parameters"
            )
        buffers_online = list(online.buffers())
        buffers_target = list(target.buffers())
        if len(buffers_online) != len(buffers_target):
            raise ValueError(
                "online and target modules must expose the same number of buffers"
            )

        self.online = online
        self.target = target
        self.m = float(momentum)
        self._params_online = params_online
        self._params_target = params_target
        self._buffers_online = buffers_online
        self._buffers_target = buffers_target

    def set_momentum(self, momentum: float) -> None:
        """Update EMA coefficient in [0,1)."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the range [0, 1)")
        self.m = float(momentum)

    @torch.no_grad()
    def copy_params(self) -> None:
        """Hard copy online params to target (for initialization)."""
        for p_t, p_o in zip(self._params_target, self._params_online):
            p_t.data.copy_(p_o.data)
        # Also copy buffers (e.g., BatchNorm running stats)
        for b_t, b_o in zip(self._buffers_target, self._buffers_online):
            b_t.data.copy_(b_o.data)

    @torch.no_grad()
    def update(self) -> None:
        """EMA update for all parameters and buffers with matching order."""
        m = self.m
        for p_t, p_o in zip(self._params_target, self._params_online):
            p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))
        for b_t, b_o in zip(self._buffers_target, self._buffers_online):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(m).add_(b_o.data, alpha=(1.0 - m))
            else:
                # For integer buffers (e.g., BatchNorm num_batches_tracked), copy
                b_t.data.copy_(b_o.data)


MomentumUpdate = MomentumUpdater


__all__ = ["MomentumUpdater", "MomentumUpdate"]
