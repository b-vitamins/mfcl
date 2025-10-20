# Fidelity Probe

The fidelity probe compares two loss implementations on the same frozen batch to
quantify behavioural drift. The probe runs only when `fidelity.enabled=true` in
the Hydra configuration. By default the feature is disabled and does not affect
training throughput.

## Configuration

Add the probe to an experiment by overriding the `fidelity` config node:

```yaml
fidelity:
  enabled: true
  loss_a: ntxent
  loss_b: ntxent
  interval_steps: 500
  beta: 1e-8
```

The `loss_a` and `loss_b` keys reference entries in `LOSS_REGISTRY`. Both loss
modules are instantiated once at startup and reused throughout training. The
`interval_steps` parameter controls how often the probe runs, while `beta`
provides a small stabiliser for the relative gradient error when norms are
nearly zero.

## Metrics

**Requirements:** your method must implement `forward_views(batch)` and return the
tensors expected by the selected losses (for example, SimCLR returns `(z1, z2)`,
BYOL returns `(p1, z2, p2, z1)`, and SwAV returns `(logits_per_crop, code_indices)`).

Every probe writes a row to `fidelity.csv` inside the training run directory
containing:

| Column        | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `step`        | Global trainer step (micro-batches processed).                   |
| `epoch`       | Epoch number at the time of measurement.                        |
| `loss_a`      | Scalar loss from the first candidate.                           |
| `loss_b`      | Scalar loss from the second candidate.                          |
| `loss_diff`   | Absolute difference `|loss_a - loss_b|`.                        |
| `grad_cos`    | Cosine similarity between gradients with respect to encoder outputs. |
| `grad_rel_norm` | Relative gradient norm error `||g_a - g_b|| / (||g_a|| + beta)`. |

All gradients are computed with encoder weights frozen to avoid mutating model
state or optimizer buffers. The probe never performs an optimizer step.
