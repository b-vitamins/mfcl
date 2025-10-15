# BYOL — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=byol
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=byol
```

## Plots

```bash
python visual.py --runs runs/byol_resnet18_160/2025...
```

**Notes**

* EMA target and predictor head; cosine or MSE variants.

