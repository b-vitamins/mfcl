"""Unit tests for mixture moment diagnostics."""

from __future__ import annotations

import csv
import itertools

import torch

from mfcl.mixture import MixtureStats


def _sample_mixture(
    *,
    pi: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sample a batch from a synthetic Gaussian mixture."""

    k = pi.numel()
    assignments = torch.multinomial(pi, num_samples=batch_size, replacement=True)
    samples = torch.zeros(batch_size, mu.size(1))
    cholesky = torch.stack([torch.linalg.cholesky(c) for c in cov])
    for component in range(k):
        mask = assignments == component
        count = int(mask.sum())
        if count == 0:
            continue
        noise = torch.randn(count, mu.size(1)) @ cholesky[component].t()
        samples[mask] = mu[component] + noise
    return samples


def _match_components(mu_est: torch.Tensor, mu_true: torch.Tensor) -> tuple[int, ...]:
    k = mu_true.size(0)
    perms = itertools.permutations(range(k))
    best_perm: tuple[int, ...] | None = None
    best_error = float("inf")
    for perm in perms:
        permuted = mu_est[list(perm)]
        error = torch.norm(permuted - mu_true).item()
        if error < best_error:
            best_error = error
            best_perm = tuple(perm)
    assert best_perm is not None
    return best_perm


def _component_covariance(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.shape[0] == 0:
        d = vectors.shape[1]
        return torch.zeros(d, d)
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / vectors.shape[0]


def test_kmeans_mixture_estimator_converges() -> None:
    torch.manual_seed(0)
    k = 3
    d = 2
    pi = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    mu = torch.tensor([[2.0, 1.0], [-1.5, 0.0], [0.0, 3.0]], dtype=torch.float32)
    cov = torch.stack(
        [
            torch.tensor([[0.4, 0.1], [0.1, 0.3]], dtype=torch.float32),
            torch.tensor([[0.2, -0.05], [-0.05, 0.4]], dtype=torch.float32),
            torch.tensor([[0.3, 0.0], [0.0, 0.2]], dtype=torch.float32),
        ]
    )

    estimator = MixtureStats(
        K=k,
        assigner="kmeans_online",
        mode="ema",
        ema_decay=0.9,
        enabled=True,
        max_assign_iters=2,
    )

    stats = {}
    for _ in range(200):
        batch = _sample_mixture(pi=pi, mu=mu, cov=cov, batch_size=512)
        stats = estimator.update(batch)

    assert stats, "Estimator returned empty statistics"
    perm = _match_components(stats["mu"], mu)

    pi_est = stats["pi"][list(perm)]
    mu_est = stats["mu"][list(perm)]
    cov_est = stats["Sigma"][list(perm)]

    assert torch.allclose(pi_est, pi, atol=0.05)
    assert torch.allclose(mu_est, mu, atol=0.2)
    assert torch.allclose(cov_est, cov, atol=0.2)

    global_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
    mu_diff = mu - global_mu.unsqueeze(0)
    B_true = torch.einsum("k,kd,ke->de", pi, mu_diff, mu_diff)
    assert torch.allclose(stats["B"], B_true, atol=0.2)
    assert stats["pi"].dtype == torch.float32
    assert stats["mu"].dtype == torch.float32
    assert stats["Sigma"].dtype == torch.float32
    assert stats["B"].dtype == torch.float32

    estimator.close()


def test_label_supervised_replicates_class_moments() -> None:
    torch.manual_seed(1)
    k = 4
    d = 3
    counts = torch.tensor([20, 15, 10, 5])
    labels = torch.repeat_interleave(torch.arange(k), counts)
    base = torch.randn(k, d)
    embeddings = torch.zeros(labels.numel(), d)
    offset = 0
    covariances = []
    for component in range(k):
        count = int(counts[component])
        samples = base[component] + 0.3 * torch.randn(count, d)
        embeddings[offset : offset + count] = samples
        covariances.append(_component_covariance(samples))
        offset += count

    estimator = MixtureStats(
        K=k,
        assigner="label_supervised",
        mode="per_batch",
        enabled=True,
    )
    stats = estimator.update(embeddings, labels)

    expected_pi = counts.float() / counts.sum()
    expected_mu = torch.stack(
        [embeddings[labels == idx].mean(dim=0) for idx in range(k)]
    )
    expected_cov = torch.stack(covariances)

    assert torch.allclose(stats["pi"], expected_pi, atol=1e-6)
    assert torch.allclose(stats["mu"], expected_mu, atol=1e-6)
    assert torch.allclose(stats["Sigma"], expected_cov, atol=1e-5)

    global_mu = (expected_pi.unsqueeze(1) * expected_mu).sum(dim=0)
    mu_diff = expected_mu - global_mu.unsqueeze(0)
    B_expected = torch.einsum("k,kd,ke->de", expected_pi, mu_diff, mu_diff)
    assert torch.allclose(stats["B"], B_expected, atol=1e-6)

    estimator.close()


def test_logging_outputs_csv(tmp_path) -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.0]], dtype=torch.float32
    )
    estimator = MixtureStats(
        K=2,
        assigner="kmeans_online",
        mode="per_batch",
        enabled=True,
        log_dir=tmp_path,
        store_scores=True,
    )

    stats = estimator.update(embeddings)
    estimator.log_step(step=3, epoch=1)
    estimator.close()

    csv_path = tmp_path / "mixture.csv"
    assert csv_path.exists()
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["step"] == "3"
    assert row["epoch"] == "1"
    assert row["K"] == "2"

    scores_path = tmp_path / "mixture_scores.pt"
    assert scores_path.exists()
    saved = torch.load(scores_path)
    assert isinstance(saved, list) and len(saved) == 1
    assert torch.equal(saved[0]["responsibilities"], stats["R"].cpu())


def test_logging_skipped_on_non_main(tmp_path) -> None:
    embeddings = torch.randn(6, 3)
    estimator = MixtureStats(
        K=2,
        assigner="kmeans_online",
        mode="per_batch",
        enabled=True,
        log_dir=tmp_path,
        store_scores=True,
        is_main=False,
    )

    stats = estimator.update(embeddings)
    estimator.log_step(step=1, epoch=0, stats=stats)
    estimator.close()

    assert not (tmp_path / "mixture.csv").exists()
    assert not (tmp_path / "mixture_scores.pt").exists()


def test_bf16_inputs_are_promoted_to_float32() -> None:
    embeddings = torch.randn(32, 5, dtype=torch.bfloat16)
    estimator = MixtureStats(
        K=3,
        assigner="kmeans_online",
        mode="per_batch",
        enabled=True,
    )

    stats = estimator.update(embeddings)

    assert stats
    assert stats["pi"].dtype == torch.float32
    assert stats["mu"].dtype == torch.float32
    assert stats["Sigma"].dtype == torch.float32
    assert stats["B"].dtype == torch.float32
