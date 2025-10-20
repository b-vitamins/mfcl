# Communication Telemetry

MFCL can attribute distributed communication volume and latency to individual
collectives and high-level payload categories. When enabled, the runtime records
per-step statistics to `comms.csv` in the run directory.

## Configuration

Communication logging is controlled via the `runtime.comms_log.enabled` flag.

```yaml
runtime:
  comms_log:
    enabled: true  # default
```

Only the main process writes `comms.csv`. The file is append-only and includes a
header row.

## Captured metrics

Each recorded step appends a row with the following schema:

| Column | Description |
| --- | --- |
| `step` | Global step identifier (1-indexed). |
| `epoch` | Epoch containing the step. |
| `world_size` | Number of distributed ranks participating in the collectives. |
| `bytes_all_reduce` | Estimated bytes transferred by `all_reduce` during the step. |
| `bytes_all_gather` | Estimated bytes transferred by `all_gather` during the step. |
| `bytes_reduce_scatter` | Estimated bytes transferred by `reduce_scatter` during the step. |
| `bytes_broadcast` | Estimated bytes transferred by `broadcast` during the step. |
| `bytes_total` | Sum of bytes across all tracked collectives. |
| `t_all_reduce_ms` | Wall-clock milliseconds spent inside `all_reduce` wrappers. |
| `t_all_gather_ms` | Wall-clock milliseconds spent inside `all_gather` wrappers. |
| `t_reduce_scatter_ms` | Wall-clock milliseconds spent inside `reduce_scatter` wrappers. |
| `t_broadcast_ms` | Wall-clock milliseconds spent inside `broadcast` wrappers. |
| `t_total_ms` | Sum of collective timings for the step. |
| `eff_bandwidth_MiBps` | Effective bandwidth (`bytes_total` converted to MiB divided by `t_total_ms`). |
| `bytes_features_allgather` | Bytes attributed to `PayloadCategory.FEATURES_ALLGATHER`. |
| `bytes_moments_mu` | Bytes attributed to `PayloadCategory.MOMENTS_MU`. |
| `bytes_moments_sigma_full` | Bytes attributed to `PayloadCategory.MOMENTS_SIGMA_FULL`. |
| `bytes_moments_sigma_diag` | Bytes attributed to `PayloadCategory.MOMENTS_SIGMA_DIAG`. |
| `bytes_mixture_muK` | Bytes attributed to `PayloadCategory.MIXTURE_MU_K`. |
| `bytes_mixture_sigmaK` | Bytes attributed to `PayloadCategory.MIXTURE_SIGMA_K`. |
| `bytes_third_moment_sketch` | Bytes attributed to `PayloadCategory.THIRD_MOMENT_SKETCH`. |
| `bytes_topr_indices` | Bytes attributed to `PayloadCategory.TOPR_INDICES`. |
| `bytes_other` | Bytes attributed to `PayloadCategory.OTHER`. |

The wire model assumes a ring algorithm:

* `all_reduce`: `2*(p-1)/p * size_bytes`
* `all_gather`: `(p-1) * size_bytes`
* `reduce_scatter`: `(p-1) * size_bytes`
* `broadcast`: `size_bytes`

where `p` is the world size and `size_bytes` is the payload size of the tensor
passed to the wrapper. Bytes are attributed to the payload category supplied
when invoking the wrapped collective.

## Usage

The `mfcl.distributed` module exposes telemetry-aware wrappers around
`torch.distributed` collectives. Each wrapper mirrors the signature of the
corresponding PyTorch function and adds a `category` keyword argument:

```python
import torch.distributed as dist

from mfcl.distributed import (
    all_reduce,
    all_gather,
    reduce_scatter,
    broadcast,
    PayloadCategory,
)

output = torch.randn(32, device=device)
all_reduce(output, category=PayloadCategory.MOMENTS_SIGMA_DIAG)

buckets = [torch.zeros_like(output) for _ in range(dist.get_world_size())]
all_gather(buckets, output, category=PayloadCategory.FEATURES_ALLGATHER)

shard = torch.zeros_like(output)
reduce_scatter(shard, buckets, category=PayloadCategory.MOMENTS_MU)

broadcast(output, src=0, category=PayloadCategory.OTHER)
```

Internally the wrappers record timing, estimate bytes transferred, and forward
the original return value. When communication logging is disabled the wrappers
behave identically to the underlying PyTorch calls. Asynchronous collectives are
not yet supported by the wrappers.
