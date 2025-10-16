# VICReg — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/vicreg_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/vicreg_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/vicreg_resnet18_160/2025...
```

**Notes**

* Invariance + variance + covariance objective; hinge on per-dim std.
  * Adjust `data.batch_size` and `data.num_workers` for your GPU/CPU. On a 12 GB 3060, start with `data.batch_size=192` at 160 px.

