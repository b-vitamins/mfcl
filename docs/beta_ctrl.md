# Beta Controller

The beta controller keeps the mixture inflation bound within a configurable
budget. When enabled via the runtime configuration it monitors the
per-step mixture statistics emitted by `MixtureStats` and rescales the
beta parameter before the next step begins. The controller never touches
optimizers or gradients and is fully gated by a feature flag.

## Configuration

```yaml
runtime:
  beta_ctrl:
    enabled: false                 # Feature flag (default off).
    target_mix_inflation_eps: 0.05 # Target bound \varepsilon_mix.
    beta_min: 3                    # Lower clamp on |beta|.
    beta_max: 12                   # Upper clamp on |beta|.
    ema_window: 50                 # EMA window for smoothing.
```

Set `train.save_dir` when enabling the controller so that telemetry can be
written to disk. The controller automatically reuses the mixture statistics
from Spec 11, so the mixture diagnostics must also be enabled for the bound
to be computed.

## Behaviour

Each step the controller pulls the most recent mixture estimates and
computes the bound

\[
|\beta| \sqrt{\tfrac{\operatorname{median}(x^\top B x)}{\pi_{\min}}}
 + \tfrac{\beta^2}{2} \Delta\Sigma_{\max} \le \varepsilon_{\text{mix}}.
\]

If the current beta would exceed the target tolerance the controller solves
for the largest admissible value, clips it to `[beta_min, beta_max]`, and
applies a one-sided EMA when the bound allows increases. Sudden reductions
are applied immediately to keep the bound satisfied.

## Logging

When enabled the controller appends `beta_ctrl.csv` alongside the other
runtime diagnostics. The CSV contains one row per step with the columns

```
step,epoch,beta_raw,beta_clipped,eps_target,eps_estimated,reason
```

The `reason` column reports why the beta changed:

- `within_target`: the proposed beta already satisfied the bound.
- `reduced_for_bound`: the controller reduced beta to keep the bound under
the target tolerance.
- `bound_vs_min`: the target tolerance is smaller than what the configured
`beta_min` allows.
- `insufficient_stats`: mixture statistics were unavailable, so the previous
beta is reused.

## Integration

The controller runs inside the trainer after `MixtureStats.update` completes.
The adjusted beta is written back to the underlying method via the optional
`set_mixture_beta` hook or by assigning a `mixture_beta` attribute. Methods
that consume beta can therefore read the updated value at the start of the
next step.
