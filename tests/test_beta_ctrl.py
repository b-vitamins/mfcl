"""Unit tests for the beta controller."""

from __future__ import annotations

import math

import torch

from mfcl.runtime.beta_ctrl import (
    BetaController,
    compute_inflation_scale,
    estimate_mixture_inflation,
)


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

    result = controller.step(stats, delta_sigma, 10.0)
    assert result.beta_applied < 10.0
    assert result.reason == "reduced_for_bound"

    scale = compute_inflation_scale(result.pi_min, result.median_xBx)
    eps = estimate_mixture_inflation(
        abs(result.beta_applied), scale, result.delta_sigma_max
    )
    assert eps <= controller.target_eps + 1e-6


def test_beta_controller_keeps_beta_when_stats_missing() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=3.0,
        beta_max=12.0,
        ema_window=4,
    )
    stats = {"pi": torch.tensor([0.25, 0.25, 0.25, 0.25]), "median_xBx": torch.tensor(0.0)}

    result = controller.step(stats, 0.0, 6.0)
    assert math.isclose(result.beta_applied, 6.0)
    assert result.reason == "insufficient_stats"


def test_beta_controller_smooths_increases() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=0.1,
        beta_max=12.0,
        ema_window=5,
    )
    pi = torch.tensor([0.2, 0.3, 0.25, 0.25])
    stats = {"pi": pi, "median_xBx": torch.tensor(2e-5)}

    first_result = controller.step(stats, torch.tensor(0.001), 10.0)
    assert first_result.reason == "reduced_for_bound"

    second_result = controller.step(stats, torch.tensor(0.0001), 12.0)
    assert second_result.reason == "reduced_for_bound"
    assert second_result.beta_applied > first_result.beta_applied
    assert second_result.beta_applied < second_result.beta_candidate


def test_beta_controller_clips_to_maximum() -> None:
    controller = BetaController(
        target_eps=0.5,
        beta_min=0.1,
        beta_max=2.0,
        ema_window=3,
    )
    stats = {
        "pi_min": torch.tensor(0.5),
        "median_xBx": torch.tensor(1.0e-4),
    }

    result = controller.step(stats, 0.0, 10.0)
    assert result.reason == "within_target"
    assert math.isclose(result.beta_clipped, controller.beta_max)
    assert result.clip == "max"


def test_apply_broadcast_updates_state() -> None:
    controller = BetaController(
        target_eps=0.05,
        beta_min=0.1,
        beta_max=12.0,
        ema_window=5,
    )

    initial = controller.step({"pi_min": 0.5, "median_xBx": 1e-4}, 0.0, 0.5)
    broadcasted = controller.apply_broadcast(0.75, initial)

    last_info = controller.last_info
    assert controller.last_beta is not None
    assert math.isclose(controller.last_beta, 0.75)
    assert math.isclose(broadcasted.beta_applied, 0.75)
    assert math.isclose(last_info["beta_applied"], 0.75)
    assert math.isclose(initial.beta_applied, controller.last_beta_raw)
