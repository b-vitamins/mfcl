# Consumer hardware presets

Three Hydra presets are provided under `configs/presets/` for common consumer
GPU setups. They can be activated by passing `presets=consumer_1x24gb`,
`consumer_1x12gb`, or `consumer_2x12gb` on the command line.

| Preset | Target hardware | Key adjustments |
| --- | --- | --- |
| `consumer_1x24gb` | Single 24GB GPU | 224 batch size (per rank), six dataloader workers, BF16 autocast, telemetry (memory/energy/comms) enabled. |
| `consumer_1x12gb` | Single 12GB GPU | 112 batch size (per rank), four dataloader workers, BF16 autocast, telemetry enabled. |
| `consumer_2x12gb` | Two 12GB GPUs | 128 batch per GPU with accumulation ×2 (effective global 512), four dataloader workers, BF16 autocast, telemetry enabled. |

All presets enable the stability sentry, memory snapshots, communication logs
and energy sampling so that training runs can surface issues with minimal manual
configuration. Adjust `data.root` as needed for your local dataset layout.

> **Note:** `data.batch_size` is specified per rank (per GPU) when running with
> DistributedDataParallel. Multiply by `world_size` × `train.accum_steps` to
> obtain the effective global batch size.
