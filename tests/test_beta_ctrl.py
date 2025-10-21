"""Unit tests for the beta controller."""

from __future__ import annotations

import math

import torch

from mfcl.runtime.beta_ctrl import (
    BetaController,
    estimate_mixture_inflation,
    solve_beta_for_mixture_inflation,
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
    beta_raw = 10.0

    result = controller.step(stats, delta_sigma, beta_raw)
    assert result.beta < beta_raw
    assert result.info["reason"] == "reduced_for_bound"

    pi_min = float(pi.min().item())
    scale = math.sqrt(2e-5 / pi_min)
    eps = estimate_mixture_inflation(abs(result.beta), scale, float(delta_sigma))
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

    result = controller.step(stats, 0.0, beta_raw)
    assert math.isclose(result.beta, beta_raw)
    assert result.info["reason"] == "insufficient_stats"


def test_beta_controller_clips_to_minimum_when_bound_tight() -> None:
    controller = BetaController(
        target_eps=0.01,
        beta_min=0.2,
        beta_max=12.0,
        ema_window=2,
    )
    stats = {"pi_min": torch.tensor(0.5), "median_xBx": torch.tensor(0.5)}
    delta_sigma = torch.tensor(0.5)
    beta_raw = 0.05

    result = controller.step(stats, delta_sigma, beta_raw)
    assert math.isclose(result.beta, controller.beta_min)
    assert result.info["reason"] == "bound_vs_min"
    assert result.info["clip"] == "min"


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
    assert first_result.info["reason"] == "reduced_for_bound"

    second_result = controller.step(stats, torch.tensor(0.0001), 12.0)
    assert second_result.info["reason"] == "reduced_for_bound"
    candidate = float(second_result.info["beta_candidate"])
    assert second_result.beta > first_result.beta
    assert second_result.beta < candidate


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


def test_solve_beta_for_mixture_inflation_handles_degenerate_stats() -> None:
    assert solve_beta_for_mixture_inflation(0.1, scale=0.0, delta_sigma=0.0) == 0.0
    assert solve_beta_for_mixture_inflation(0.1, scale=0.2, delta_sigma=0.0) == 0.5
    expected = math.sqrt(2 * 0.3 * 0.1) / 0.3
    assert math.isclose(
        solve_beta_for_mixture_inflation(0.1, scale=0.0, delta_sigma=0.3),
        expected,
        rel_tol=1e-6,
    )
