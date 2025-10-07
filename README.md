# MFCL

Mean-field contrastive learning utilities and baselines.

## Quickstart

```bash
# install
pip install -e .

# sanity: CLI help
mfcl --help

# run a minimal fidelity sweep (edit config paths first)
mfcl run fidelity --config configs/eval/fidelity_minimal.yaml
```

## Layout

* `mfcl/` package with CLI and modules
* `configs/` YAML configs
* `results/` artifacts (CSV/JSONL/figures)

