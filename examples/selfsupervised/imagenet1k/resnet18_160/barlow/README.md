# Barlow Twins — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/barlow_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/barlow_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/barlow_resnet18_160/2025...
```

**Notes**

* Cross-correlation matrix matches identity; off-diagonal penalty λ.
* Adjust `data.batch_size` and `data.num_workers` to fit your GPU/CPU budget. On a 12 GB RTX 3060, start with `data.batch_size=192` for 160 px crops.

