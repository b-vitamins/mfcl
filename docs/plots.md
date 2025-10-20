# Plotting utilities

The `tools/` directory now ships lightweight scripts for rendering the most
common telemetry plots from CSV exports. Each script accepts an `--input`
argument pointing at a CSV file and writes both PDF and PNG artifacts using a
consistent house style.

| Script | Expected columns | Description |
| --- | --- | --- |
| `tools.plot_runtime_vs_batch` | `batch_size`, `t_*_ms`, `comm_bytes` | Stacked runtime breakdown per batch size with a secondary axis for communication volume. |
| `tools.plot_accuracy_vs_time` | `minutes`/`time_min`, `top1`/`accuracy`, optional `ci95` or `lower`/`upper` | Iso-time accuracy curves with 95% confidence intervals. |
| `tools.plot_memory_vs_batch` | `batch_size`, `peak_gb`/`max_gb` | Peak memory consumption as the batch size changes. |
| `tools.plot_energy_per_image` | `method`, `images_per_s`/`throughput`, `energy_mj`/`energy_per_image_mj` | Energy per image versus throughput scatter plot with inline labels. |
| `tools.plot_fidelity_vs_tau` | `tau`, `fidelity`/`distance`, optional `ci95` | Fidelity sweeps over teacher momentum (tau) with error bars. |

Example usage:

```bash
python -m tools.plot_runtime_vs_batch --input runs/sim/timings_summary.csv --output-dir figures
python -m tools.plot_accuracy_vs_time --input results/accuracy_curve.csv
```

All scripts use a non-interactive Matplotlib backend and produce both
`<prefix>.pdf` and `<prefix>.png` in the requested output directory.
