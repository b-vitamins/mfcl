# SwAV — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/swav_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/swav_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/swav_resnet18_160/2025...
```

**Notes**

* Multi-crop with Sinkhorn assignments and prototypes.
  * Adjust `data.batch_size` and `data.num_workers` for your hardware. On a 12 GB RTX 3060, start with `data.batch_size=192` at 160 px.

