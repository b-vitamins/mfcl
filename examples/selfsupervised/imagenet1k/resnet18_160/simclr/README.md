# SimCLR — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/simclr_resnet18_160/2025...
```

**Notes**

* Two-view NT-Xent with color jitter, blur, grayscale.

