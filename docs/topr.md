# Top-R Mixture Diagnostics

The Top-R diagnostics module quantifies how much responsibility mass is retained
when only the R highest-weighted mixture components are kept per anchor. The
module can be enabled through the runtime mixture configuration and produces an
append-only `topr.csv` alongside other telemetry outputs.

## Configuration

```yaml
runtime:
  mixture:
    enabled: false        # Master feature flag (default off).
    K: 8                  # Number of mixture components tracked by MixtureStats.
    topR: 0               # Number of components retained per anchor (0 = measure only).
    pi_floor: 1.0e-3      # Minimum mixture weight when computing responsibilities.
    K_subsample: null     # Optional chunk size for memory-friendly processing.
```

When `runtime.mixture.enabled` is `true` the trainer instantiates both the
`MixtureStats` estimator and the `TopRDiagnostics` tracker. Each step updates the
diagnostics using the per-sample responsibilities estimated from the batch. The
tracker computes:

* **Missed responsibility mass** `ε`: the portion of probability mass outside
  the Top-R components per anchor.
* **Gradient error bound**: an upper bound on the gradient perturbation induced
  by discarding the tail components, based on batch estimates of
  `D_μ = max‖μ_k − μ_ℓ‖` and `D_Σ = max‖Σ_k − Σ_ℓ‖_{op}`.

The CSV contains per-step quantiles of both metrics:

```text
step,epoch,R,epsilon_p50,epsilon_p90,err_bound_p50,err_bound_p90
```

All computations run under `torch.no_grad()` and never alter gradients or the
training objective. When `K_subsample` is set the responsibilities are processed
in component chunks to avoid materializing full `[N×K]` tensors for large K.
