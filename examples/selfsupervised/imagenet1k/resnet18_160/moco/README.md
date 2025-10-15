# MoCo v2 — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/moco_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/moco_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/moco_resnet18_160/2025...
```

**Notes**

* Momentum encoder with queue; InfoNCE temperature ~0.2.
* Logged metrics include `loss`, `pos_sim`, `neg_sim_mean`, `queue_len`, and `queue_capacity`.

