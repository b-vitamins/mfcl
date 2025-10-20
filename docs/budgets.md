# Runtime Budgets

The runtime budget tracker allows you to run experiments under comparable
compute, token, or epoch budgets. The subsystem is disabled by default and can
be enabled via `runtime.budget.enabled` in the Hydra configuration. When
enabled, the trainer enforces the configured limits and writes a summary to
`budget.json` inside the run directory.

## Configuration

Configure budgets in `configs/runtime/default.yaml` (or an override file):

```yaml
budget:
  enabled: true
  mode: iso_tokens        # iso_time | iso_tokens | iso_epochs | comm_cap | energy_cap
  max_tokens: 50_000      # required for iso_tokens
  max_minutes: null       # required for iso_time
  max_epochs: null        # required for iso_epochs
  max_comm_bytes: null    # optional hard cap in any mode
  max_energy_Wh: null     # optional hard cap in any mode
  steps_per_epoch: null   # optional override when loader length is unknown
```

Key points:

- **Feature flag** – budgets remain opt-in. Leave `enabled: false` to revert to
  the previous behaviour.
- **Mode-specific limits** – each mode requires its corresponding limit. For
  example, `iso_tokens` needs `max_tokens`, and `iso_time` needs `max_minutes`.
- **Auxiliary caps** – you may specify `max_comm_bytes` and/or `max_energy_Wh`
  in any mode. These caps are enforced alongside the primary budget.
- **Epoch budgets** – `iso_epochs` requires the trainer to know the number of
  steps per epoch. When the dataloader has a finite length, this is inferred
  automatically. If the loader is unbounded, provide `steps_per_epoch` via the
  budget limits.

## Behaviour

- The trainer updates the tracker every training step with wall-clock time,
  total communication volume, per-step energy delta (when telemetry is
  enabled), and an accurate token count that includes all augmented views. This
  keeps token accounting correct even when using gradient accumulation.
- Before launching a step the tracker runs a pre-check when it can predict usage
  (tokens/epochs). For time, comms, and energy limits, enforcement happens right
  after the step when the true values are known. Evaluation hooks consult the
  tracker beforehand and are skipped if they would overshoot the remaining
  budget.
- Checkpoints now embed the current budget snapshot. Resuming from such a
  checkpoint preserves the remaining budget so experiments stay comparable.
- When the run finishes, a human-readable `budget.json` containing the mode,
  limits, and cumulative totals is written to the output directory.

## Monitoring

The snapshot returned by the tracker exposes:

- `tokens` – number of augmented views processed.
- `time_ms` / `time_minutes` – cumulative wall-clock training time.
- `comm_bytes` – aggregate communication volume across collectives.
- `energy_Wh` – integrated energy consumption (requires energy telemetry).
- `steps` and `epochs` – number of micro-batches processed and the equivalent
  epochs (when available).

You can inspect the JSON file or access the in-memory tracker via
`mfcl.runtime.budget.BudgetTracker` for more advanced automation or logging.
