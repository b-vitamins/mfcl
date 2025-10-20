import csv

import torch

from mfcl.mixture.topr import TopRDiagnostics, select_topR


def test_epsilon_monotonic_with_R() -> None:
    torch.manual_seed(0)
    batch, components = 64, 6
    Q = torch.randn(batch, components)
    peak_idx = torch.randint(0, components, (batch,))
    Q[torch.arange(batch), peak_idx] += 6.0
    pi = torch.abs(torch.randn(components))
    pi = pi / pi.sum()

    epsilons = []
    for R in range(1, components):
        selection = select_topR(Q, pi, R)
        epsilons.append(selection.epsilon)
    for prev, nxt in zip(epsilons, epsilons[1:]):
        assert torch.all(prev >= nxt - 1e-6)


def test_select_topr_logsumexp_stability() -> None:
    torch.manual_seed(1)
    Q = torch.randn(12, 5)
    pi = torch.ones(5) / 5.0

    result_normal = select_topR(Q, pi, 3)
    result_scaled = select_topR(750.0 * Q, pi, 3)
    assert torch.isfinite(result_scaled.epsilon).all()
    assert torch.all(result_scaled.epsilon >= 0)
    assert torch.all(result_scaled.epsilon <= 1)


def test_topr_with_R_zero_measures_full_mass() -> None:
    torch.manual_seed(2)
    Q = torch.randn(10, 4)
    pi = torch.ones(4) / 4.0
    result = select_topR(Q, pi, 0)
    assert torch.all(result.epsilon <= 1e-5)


def test_topr_diagnostics_logging(tmp_path) -> None:
    monitor = TopRDiagnostics(
        R=2,
        enabled=True,
        log_dir=tmp_path,
        is_main=True,
        pi_floor=1e-3,
    )
    responsibilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.6, 0.25, 0.15]],
        dtype=torch.float32,
    )
    pi = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float32)
    mu = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=torch.float32,
    )
    Sigma = torch.stack(
        [
            torch.eye(2, dtype=torch.float32),
            torch.diag(torch.tensor([1.1, 0.9])),
            torch.diag(torch.tensor([0.8, 1.2])),
        ]
    )
    metrics = monitor.update(
        responsibilities=responsibilities,
        pi=pi,
        mu=mu,
        Sigma=Sigma,
        beta=2.0,
    )
    assert metrics and "epsilon" in metrics and "err_bound" in metrics
    monitor.log_step(step=1, epoch=0)
    monitor.close()

    csv_path = tmp_path / "topr.csv"
    assert csv_path.exists()
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["R"] == "2"
    assert "epsilon_p50" in row and "err_bound_p90" in row
