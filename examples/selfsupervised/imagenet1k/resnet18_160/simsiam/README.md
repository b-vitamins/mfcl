# SimSiam — ResNet-18 @ 160 px on ImageNet-1K

## Train
```bash
python train.py --config-name experiments/simsiam_r18_160 data.root=/path/to/imagenet
```

## Linear eval

```bash
MFCL_CKPT=/path/to/ckpt_ep0200.pt \
python evallinear.py --config-name experiments/simsiam_r18_160 data.root=/path/to/imagenet
```

## Plots

```bash
python visual.py --runs runs/simsiam_resnet18_160/2025...
```

**Notes**

* Predictor head, stop-grad on target branch.
* Adjust `data.batch_size` and `data.num_workers` to fit your hardware. On a 12 GB 3060, start with `data.batch_size=192` for 160 px.

