"""Unit tests for the beta controller."""

from __future__ import annotations

import math

import torch

from mfcl.runtime.beta_ctrl import BetaController


def test_beta_controller_reduces_beta_when_bound_exceeded() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=0.1,
        beta_max=12.0,
        ema_window=5,
    )
    pi = torch.tensor([0.2, 0.3, 0.25, 0.25])
    stats = {"pi": pi, "median_xBx": torch.tensor(2e-5)}
    delta_sigma = torch.tensor(0.001)
    beta_raw = 10.0

    beta_clipped, info = controller.step(stats, delta_sigma, beta_raw)
    assert beta_clipped < beta_raw
    assert info["reason"] == "reduced_for_bound"

    pi_min = float(pi.min().item())
    scale = math.sqrt(2e-5 / pi_min)
    eps = abs(beta_clipped) * scale + 0.5 * (beta_clipped ** 2) * float(delta_sigma)
    assert eps <= controller.target_eps + 1e-6


def test_beta_controller_keeps_beta_when_stats_missing() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=3.0,
        beta_max=12.0,
        ema_window=4,
    )
    stats = {"pi": torch.tensor([0.25, 0.25, 0.25, 0.25]), "median_xBx": torch.tensor(0.0)}
    beta_raw = 6.0

    beta_clipped, info = controller.step(stats, 0.0, beta_raw)
    assert math.isclose(beta_clipped, beta_raw)
    assert info["reason"] == "insufficient_stats"


def test_beta_controller_smooths_increases() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=0.1,
        beta_max=12.0,
        ema_window=5,
    )
    pi = torch.tensor([0.2, 0.3, 0.25, 0.25])
    stats = {"pi": pi, "median_xBx": torch.tensor(2e-5)}

    first_beta, info = controller.step(stats, torch.tensor(0.001), 10.0)
    assert info["reason"] == "reduced_for_bound"

    second_beta, info2 = controller.step(stats, torch.tensor(0.0001), 12.0)
    assert info2["reason"] == "reduced_for_bound"
    candidate = float(info2["beta_candidate"])
    assert second_beta > first_beta
    assert second_beta < candidate


def test_apply_broadcast_updates_state() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=0.1,
        beta_max=12.0,
        ema_window=5,
    )

    info = {"beta_clipped": 0.75, "reason": "broadcasted"}
    controller.apply_broadcast(0.75, info)

    last_info = controller.last_info
    assert controller.last_beta is not None
    assert math.isclose(controller.last_beta, 0.75)
    assert last_info["beta_clipped"] == 0.75
    assert last_info["reason"] == "broadcasted"
    assert last_info["beta_applied"] == 0.75
    assert "beta_applied" not in info
