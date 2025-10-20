# Memory Telemetry & OOM Search

MFCL provides opt-in GPU memory instrumentation and a helper script for
identifying stable batch sizes.

## Runtime telemetry

The memory monitor is guarded by `runtime.memory.enabled` (default `true`). When
enabled, the trainer records two sources of information. Note the **semantics**:
`event="step"` rows reflect **per-process** values from PyTorch; `event="nvml"`
rows report **device-wide** totals from NVML.

- A 1 Hz NVML sampler capturing device-wide `mem_used_MB`. Optional placeholders
  are included for
  `mixture_buffers_MB`, `resp_buffer_MB` and `sketch_buffer_MB` so downstream
  tooling can attach custom buffer accounting.
- Step-level snapshots (event=`"step"`) of **per-process**
  `torch.cuda.memory_stats()` captured every
  `runtime.memory.step_interval` steps (default `10`).

All entries are appended to `<save_dir>/memory.csv` alongside `step` and `epoch`
metadata. When the feature flag is disabled, no additional work is performed.

## Batch-size search

`tools/oom_search.py` performs a binary search over `train.batch_size` to find
the largest configuration that can run a fixed number of steps without raising
an OOM. The search can optionally sweep over `loss.covariance_mode`,
`mixture.K` and `mixture.topR` combinations.

Example usage:

```bash
python tools/oom_search.py --min-batch 64 --max-batch 512 --steps 8 \
  --search-covariance --search-mixture --mixture-values 8,4 12,6
```

Successful runs emit two artifacts in the chosen output directory:

- `oom_search_summary.json` – structured summary (best overrides, throughput,
  peak memory).
- `oom_override.yaml` – Hydra-compatible override file to re-run training with
  the recommended settings.

The script uses real training runs when executed normally, constraining each
trial to the requested step budget via the `MFCL_OOM_SEARCH_MAX_STEPS`
environment variable. Unit tests rely on the `MFCL_FAKE_OOM_THRESHOLD`
environment variable for deterministic behaviour.
