# BYOL — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/byol_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/byol_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/byol_resnet18_160/2025...
```

**Notes**

* EMA target and predictor head; cosine or MSE variants.
* Adjust `data.batch_size` and `data.num_workers` to fit your GPU/CPU. On a 12 GB 3060, start with `batch_size=192` for 160 px.

