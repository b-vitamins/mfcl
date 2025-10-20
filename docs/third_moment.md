# Third-Moment Sketch (κ₃)

The third-moment sketch provides a lightweight diagnostic for measuring
skewness in the embedding space. Rather than materialising the full third
order tensor, MFCL draws a fixed set of random projection directions and keeps
an exponential moving average of the projected third central moments.

When enabled (`runtime.third_moment.enabled=true`) the trainer instantiates a
`ThirdMomentSketch` object that:

* draws a seeded projection matrix `R ∈ ℝ^{d×r}` with unit-norm columns,
* requires an external estimate of the embedding mean `μ` (from Spec 11) via
  `ThirdMomentSketch.set_mean`,
* updates the projected third moments using distributed all-reduces over the
  `r` sketch coordinates only, and
* clips per-anchor estimates at `±5 × MAD` before summarising their absolute
  value.

During each training step the SimCLR loss obtains the active sketch and, when a
mean is available, feeds the current mini-batch projections through it. The
trainer then appends a row to `kappa3.csv` containing the median (P50) and
90th percentile (P90) of `|κ₃(x)|` for the anchors processed on that rank.
The CSV is append-only with the following columns:

| column | description |
| ------ | ----------- |
| `step` | Global training step (micro-batches). |
| `epoch` | Epoch index when the row was recorded. |
| `p50_abs` | Median of `|κ₃(x)|` after MAD clipping. |
| `p90_abs` | 90th percentile of `|κ₃(x)|` after MAD clipping. |

Only `Θ(r)` scalars are communicated per step (the projected moment sums plus a
count), keeping the overhead well within the diagnostic budget.

## Configuration

```yaml
runtime:
  third_moment:
    enabled: true
    rank: 16       # number of sketch directions (diagnostic cost scales linearly)
    seed: 41       # deterministic projection matrix seed
    ema_decay: 0.95
```

A non-null `train.save_dir` is recommended so `kappa3.csv` can be written. The
sketch is inert unless explicitly enabled.

## Usage notes

* Call `ThirdMomentSketch.set_mean(mu)` with the global embedding mean produced
  by the moments subsystem (Spec 11) before the first update. Updates are
  skipped if the mean is unavailable.
* `ThirdMomentSketch.update` can optionally accept a batch-specific mean or
  anchor directions; by default it uses the centred embeddings as anchors.
* Use `ThirdMomentSketch.estimate(x)` to query `κ₃(x)` for arbitrary directions
  (useful in tests or offline diagnostics).
