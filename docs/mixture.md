# Mixture Diagnostics

The mixture diagnostics module maintains per-step estimates of a K-component
Gaussian mixture. When enabled through the runtime configuration the estimator
tracks:

- Mixing weights pi_k
- Component means mu_k
- Component covariances Sigma_k
- Between-component covariance matrix B

## Configuration

```yaml
runtime:
  mixture:
    enabled: false          # Feature flag, default disabled.
    K: 8                    # Number of mixture components.
    mode: ema               # Accumulator mode: per_batch | ema | label_supervised.
    assigner: kmeans_online # Assignment strategy: kmeans_online | label_supervised.
    ema_decay: 0.95         # EMA decay factor when mode == "ema".
    store_scores: false     # Persist responsibilities alongside CSV output.
    scores_mode: append     # append | per_step storage strategy for responsibilities.
    cross_rank: false       # Gather embeddings across ranks before computing stats.
    max_assign_iters: 2     # Lloyd iterations for the k-means assigner.
```

With `assigner=kmeans_online` the module runs a small number of Lloyd
iterations (warm-started from previous centroids) to obtain per-sample
responsibilities. When `assigner=label_supervised` the provided labels map
directly to components.

## Logging

When the feature flag is enabled and a `log_dir` is supplied the estimator
appends a `mixture.csv` file containing one row per training step. Only the
main rank writes to disk to avoid corruption in distributed runs:

```
step,epoch,K,pi_min,pi_max,trace_B,opnorm_B
```

If `store_scores=True` the per-sample responsibilities are stored alongside the
CSV. The default `scores_mode=append` maintains a single `mixture_scores.pt`
file containing a list of per-step entries. Set `scores_mode=per_step` to emit
individual shards (`mixture_scores/step_000123.pt`) instead, which is safer for
long jobs.

## Usage

```python
from mfcl.mixture import MixtureStats

estimator = MixtureStats(
    K=8,
    assigner="kmeans_online",
    mode="ema",
    enabled=True,
    log_dir=run_dir,
    is_main=True,
)
stats = estimator.update(embeddings)
# stats contains pi, mu, Sigma, B, and the per-sample responsibilities R.
estimator.log_step(step=step_idx, epoch=current_epoch, stats=stats)
```

All diagnostics are computed under `torch.no_grad()` in float32, so supplying
bf16/half activations is safe. The estimator never interacts with optimizers or
gradients. When enabled via the runtime configuration the training loop exposes
the estimator through a lightweight context; the SimCLR loss automatically feeds
embeddings into the estimator each step.
