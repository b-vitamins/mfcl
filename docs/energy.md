# Energy Telemetry

MFCL can record GPU power draw and derive energy consumption metrics during
training. The feature is disabled by default and requires NVIDIA's NVML
libraries to be available and accessible to the training process.

## Enabling

Set the runtime energy configuration flag when launching training, for example:

```bash
python train.py runtime.energy.enabled=true
```

Optional overrides:

* `runtime.energy.kwh_price_usd` — electricity price used to estimate the USD
  cost of each epoch (default: `0.25`).
* `runtime.energy.sample_interval_s` — polling interval in seconds for the NVML
  power sampler (default: `1.0`).

Only the main process writes telemetry. If NVML is unavailable or permission is
denied the monitor logs a warning and disables itself without interrupting
training. Readings are taken for **GPUs visible to the process**. In typical
single-process-per-GPU DDP deployments, rank 0 therefore logs a single GPU.

## Output

When enabled a CSV named `energy.csv` is created inside the run directory. The
file is append-only and contains per-sample power and energy readings for each
visible GPU at ~1 Hz:

```
timestamp,step,epoch,gpu_id,power_W,energy_Wh_cum,energy_J_cum
```

Energy columns are cumulative per GPU. Joules are derived directly from the
integrated power samples (1 W = 1 J/s); watt-hours are stored for convenience
and potential cost calculations. The cumulative totals are monotonically
non-decreasing.

## Training summary

At the end of each epoch, the console summary prints:

* `energy_epoch_Wh` — total watt-hours consumed across all **visible** GPUs during
  the epoch.
* `energy_per_image_J` — joules per image based on the global sample count.
* `energy_epoch_cost_usd` — optional electricity cost estimate when
  `runtime.energy.kwh_price_usd` is greater than zero.

These metrics also appear in the dictionary returned from
`Trainer.train_one_epoch` for downstream consumers.
