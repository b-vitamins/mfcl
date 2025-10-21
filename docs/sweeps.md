# Sweep Automation

This document describes the Spec 9 sweep harness responsible for launching
scaling/fairness experiments and collating their results. The feature set is
guarded behind explicit flags so existing workflows remain unchanged.

## Running a Sweep

`tools/sweep.py` expands a YAML (or JSON-compatible) grid into concrete runs and
launches them via `torchrun`. The script is disabled by default – pass
`--enable-sweeps` to acknowledge the feature flag.

```bash
python -m tools.sweep --enable-sweeps path/to/grid.yaml
```

Key points:

- The grid file exposes a list of `grid` entries. Each entry defines a set of
  `params` that may be scalars or lists (expanded via Cartesian product).
- Common convenience aliases map to Hydra overrides. `model` and `method`
  select Hydra groups via `model=<name>` / `method=<name>` (no JSON quoting),
  while scalars like `batch_size` continue to map to concrete config keys such
  as `data.batch_size`. The `amp_dtype` alias toggles `train.loss_fp32` and
  exposes `MFCL_AMP_DTYPE` to downstream consumers.
- Budget definitions automatically enable the runtime budget tracker and set
  the relevant limits (e.g., `mode: iso_time` with `limit: 2.5` writes
  `runtime.budget.max_minutes=2.5`). Optional fields `target_metric` and
  `target_value` are recorded for downstream reporting.
- Each run receives a deterministic identifier (`<index>_<group>[_<combo>]`),
  a dedicated output directory, and environment metadata exposed through
  `MFCL_SWEEP_*` variables for custom hooks. When `MASTER_PORT` is not provided
  the runner auto-selects a free local port to avoid collisions across
  back-to-back launches.
- Multi-node launches can be configured by specifying `nnodes`, `node_rank`,
  and an optional `rendezvous` block (`backend`, `endpoint`, `id`, `conf`) at
  the grid root or per-entry level. These values drive the generated
  `torchrun` flags (`--nnodes`, `--rdzv_backend`, `--rdzv_endpoint`, etc.) and
  populate `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `NNODES`, and `NODE_RANK` in the
  child environment.
- The sweep manifest (`sweep_manifest.json`) captures the full configuration,
  overrides, environment, and completion status of every run.

Example grid: see `configs/sweeps/example_iso_time.yaml`.

## Aggregating Results

`tools/aggregate.py` ingests the manifest alongside individual run artifacts
(`timings.csv`, `comms.csv`, `budget.json`, optional eval/energy/fidelity CSVs)
to produce a machine-readable summary and a Markdown report. As with the grid
runner, the tool is feature flagged via `--enable-aggregator`.

```bash
python -m tools.aggregate --enable-aggregator --root /path/to/sweep
```

The aggregator will fail fast if a completed run is missing a required CSV and
marks any unsuccessful runs with `status=error` to avoid silent drops. Summary
columns include:

- `ips_mean`, `step_time_ms`, and `bytes_per_step` derived from telemetry
- `accuracy_value` and optional `time_to_target_min` for the configured
  `target_metric`
- Budget metadata (`budget_mode`, limits, cumulative totals)
- Optional diagnostics (`energy_Wh`, `fidelity_mean`)

Reports are written to `<root>/reports/summary_<timestamp>.csv` and
`<root>/reports/summary.md` by default.

## Testing

`pytest tests/test_sweep.py` exercises the end-to-end harness using a dry-run
stub that writes deterministic telemetry files. Complementary coverage in
`tests/integration/test_sweep.py` ensures multi-node grid options are parsed
into the expected launcher arguments and environment variables.
