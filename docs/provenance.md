# Provenance Manifests

The provenance subsystem captures a detailed, tamper-resistant manifest for
every training and evaluation run. When `runtime.provenance` is enabled (the
default), the runner writes three artifacts inside the run's
`provenance/` directory:

- `repro.json`: canonical JSON manifest containing snapshots of the
  configuration, environment, hardware, RNG seeds, and dataset fingerprints.
  The file stores a `history` list so resumed runs append new snapshots without
  mutating prior entries.
- `git.diff`: the working tree diff captured at the time of collection.
  Clean repositories produce an empty file while dirty trees record the full
  patch.
- `env.txt`: snapshot of the process environment variables sorted by key.

## Captured fields

Each snapshot returned by `mfcl.utils.provenance.collect_provenance` includes
these sections:

- **git**: commit SHA, porcelain status, remote URL (if available),
  working-tree diff, and a `dirty` flag.
- **runtime**: Python and library versions (PyTorch, torchvision, CUDA,
  cuDNN, NCCL), AMP dtypes, and determinism flags.
- **hardware**: host identifiers, GPU inventory and driver version, CPU
  model, RAM estimate, and distributed ranks.
- **seeds**: Python, NumPy, PyTorch CPU/GPU seeds plus the dataloader worker
  policy and base seed.
- **dataset**: SHA256 digest of the class-index file when present and up to
  1,000 hashed dataset identifiers (relative paths only).
- **loss/diagnostics**: placeholders for covariance, moment estimators, and
  advanced diagnostic toggles to simplify future extensions.

Snapshots also embed the resolved Hydra configuration under `run_config` so the
exact overrides used for a run remain available.

## Integration points

`train.py` and `eval.py` call `collect_provenance` during startup. When
resuming from a checkpoint, the runners append an event of the form
`{"type": "resume", "resumed_from": "/path/to/ckpt.pt", "time": ...}` to the
snapshot before persisting. Evaluation runs mark `program="eval"` while
training runs mark `program="train"`.

## Configuration

The new Hydra group `runtime` exposes a boolean flag:

```yaml
# configs/runtime/default.yaml
provenance: true
```

Disable provenance for performance-sensitive debugging by launching with
`runtime.provenance=false`.

## Deterministic manifests

When the repository is clean and the configuration is unchanged, the contents
of `provenance/repro.json` are stable between runs. Any working-tree
modification sets the `git.dirty` flag and populates `git.diff`, ensuring
changes are always auditable.
