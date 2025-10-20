# Stress Diagnostics

MFCL ships optional stress-test utilities for probing representation robustness.
All features are disabled by default and gated behind explicit configuration
flags to avoid changing training behavior unless requested.

## Class-Packed Sampler

Enable `data.class_packed.enabled=true` to draw balanced batches with a fixed
number of classes and examples per class. When enabled the training batch size
must equal `num_classes_per_batch * instances_per_class`.

```yaml
# Example override
{data:
  class_packed: {
    enabled: true,
    num_classes_per_batch: 16,
    instances_per_class: 2,
    seed: 0,
  }
}
```

Distributed training is supported; each rank receives whole batches preserving
class balance.

## Hardness Telemetry

The hardness monitor samples negative similarities each step, records the P50
and P90 of top-1/top-5 negatives, and appends rows to
`$RUN_DIR/hardness.csv`. Enable it via the runtime config:

```yaml
runtime:
  hardness:
    enabled: true
    sample_anchors: 512      # max anchors per step
    sample_negatives: 4096   # optional cap on negatives per anchor (aka max_negatives)
    topk: [1, 5]
```

The monitor integrates with InfoNCE-style losses (SimCLR, MoCo) and the trainer
without altering gradients. You can also set `max_negatives`, an alias for
`sample_negatives`. Overhead remains below the global telemetry budget (<5%).

## Synthetic Stress Rigs

The `mfcl.stress` package provides utilities for offline diagnostics:

* `generate_clustered_embeddings` — draws unit-norm embeddings from a
  K-Gaussian mixture, optionally writing embeddings and the ground-truth
  centroids/assignments to disk for validation.
* `inject_heavy_tails` — replaces a configurable fraction of similarity scores
  with samples from a symmetric heavy-tailed mixture for adversarial testing.

Both tools are configured under the `stress` config group for scripted usage:

```yaml
stress:
  synthetic_clusters:
    enabled: false
    k: 8
    scale: 0.2
    samples: 8192
    dim: 128
    seed: 0
    output_dir: null
  heavy_tail:
    enabled: false
    p_tail: 0.1
    tail_scale: 5.0
```

Scripts can import `mfcl.stress` to programmatically generate clustered
embeddings or inject heavy-tailed perturbations into similarity tensors.
