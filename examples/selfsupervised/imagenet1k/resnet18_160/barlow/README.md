# Barlow Twins — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=barlow
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=barlow
```

## Plots

```bash
python visual.py --runs runs/barlow_resnet18_160/2025...
```

**Notes**

* Cross-correlation matrix matches identity; off-diagonal penalty λ.

