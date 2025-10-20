# Telemetry and Timing

MFCL provides per-step timing instrumentation that attributes wall-clock time to
major training phases. When enabled, the trainer records timings to
`timings.csv` in the run directory and can optionally emit NVTX ranges for GPU
profiling.

## Configuration

Telemetry is controlled by the `runtime.timing` and `runtime.nvtx` configuration
nodes:

```yaml
runtime:
  timing:
    enabled: true       # master switch for timing capture
    warmup_steps: 50    # skip the first N steps before recording
    sample_rate: 1      # record every k-th step after warmup
  nvtx: false           # wrap segments in NVTX ranges when true
```

Only the main process writes `timings.csv`. Non-zero `sample_rate` values reduce
recording frequency while still keeping the CUDA event objects warm to minimise
allocation overhead.

## Timed segments and output

Each recorded step appends a row with the following schema:

| Column | Description |
| --- | --- |
| `step` | Global step number (1-indexed). |
| `epoch` | Epoch containing the step. |
| `t_data_ms` | Dataloader + host-to-device transfer time in milliseconds. |
| `t_fwd_ms` | Forward pass (CUDA event) latency in milliseconds. |
| `t_bwd_ms` | Backward pass latency in milliseconds. |
| `t_opt_ms` | Optimizer step latency in milliseconds. |
| `t_comm_ms` | Communication time in milliseconds (populated by comms logger). |
| `t_assign_ms` | Placeholder timer for mixture assignments. |
| `t_topr_ms` | Placeholder timer for top-R selection. |
| `t_beta_ctrl_ms` | Placeholder timer for beta controller updates. |
| `t_misc_ms` | Remainder of the step (hooks, logging, diagnostics). |
| `t_step_ms` | Total wall-clock step duration. |
| `ips_step` | Images-per-second estimate for the step. |

CUDA events are reused across steps and only synchronised when sampling a step,
which keeps the runtime overhead under ~2% (see `tests/test_timers.py`).

## NVTX integration

Setting `runtime.nvtx=true` wraps each timed segment in an NVTX range. Open the
resulting Chrome trace (for example, via Nsight Systems) to validate segment
boundaries and correlate CSV metrics with GPU execution timelines.

## Extending telemetry

The `StepTimer` exposes helpers such as `range_assign`, `range_topr`,
`range_beta_ctrl`, and `range_misc` for future diagnostics. Additional systems
(e.g. Spec 3 communications logging) can call `set_comm_ms`/`add_comm_ms` on the
trainer's `step_timer` instance to contribute communication timings.
