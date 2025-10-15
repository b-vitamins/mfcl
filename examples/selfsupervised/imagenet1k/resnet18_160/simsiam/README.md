# SimSiam — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=simsiam
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/simclr_r18_160 data.root=/path/to/imagenet method=simsiam
```

## Plots

```bash
python visual.py --runs runs/simsiam_resnet18_160/2025...
```

**Notes**

* Predictor on both views; stop-grad handled in loss.

