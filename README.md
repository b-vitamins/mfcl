# mfcl

Mean-field contrastive learning (MFCL) ships a unified, Hydra-driven
configuration stack for ImageNet-style self-supervised learning. The
repository exposes two entrypoints – `train.py` for pretraining and
`eval.py` for frozen linear evaluation – together with lean
utilities for monitoring and plotting runs.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional extras (e.g. Apex, timm) only if you explicitly need
them. All commands below assume they are executed from the project
root with the virtual environment activated.

## ImageNet layout

Arrange your ImageNet-1K data as an `ImageFolder` directory before
launching training:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

The default data config points to
`$HOME/.cache/torch/datasets/imagenet/ILSVRC2012_img`; override via
`--dataset-root /custom/path` or by setting `IMAGENET_ROOT` in the
environment.【F:configs/data/imagenet.yaml†L1-L11】

## Training interface

`train.py` layers a compact CLI on top of Hydra overrides so you can
mix ergonomic flags with advanced overrides when needed.【F:train.py†L111-L216】
Inspect the full option set with `python train.py --help`. Common flags
include:

| Flag | Purpose |
| --- | --- |
| `--method/-m` | Choose SSL method (simclr, moco, byol, simsiam, swav, barlow, vicreg). |
| `--model/-b` | Select encoder config (resnet18/34/50 or `timm:<name>`). |
| `--data/-d` | Switch between `imagenet` and `synthetic` loaders. |
| `--optimizer/-o` | Pick optimizer preset (`sgd`, `lars`, `adamw`). |
| `--batch-size` / `--num-workers` | Control dataloader settings. |
| `--img-size` / `--augment` | Override augmentation pipeline. |
| `--epochs` / `--warmup` | Adjust schedule length. |
| `--lr`, `--weight-decay`, `--momentum`, `--beta1`, `--beta2` | Tune optimizer hyperparameters. |
| `--knn` / `--knn-period` | Enable periodic kNN evaluation during training. |
| `--run-dir` | Force checkpoints into a specific directory. |

Additional overrides can be appended in Hydra syntax (e.g.
`method.temperature=0.2`).

### Quick synthetic shakeout

The synthetic preset relies on `torchvision.datasets.FakeData`, letting
you validate the full stack in seconds.【F:configs/data/synthetic.yaml†L1-L14】
Example command:

```bash
python train.py --method simclr --data synthetic --model resnet18 \
  --batch-size 8 --epochs 1 --lr 0.01 --num-workers 0 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 \
  hydra.run.dir=./tmp_runs/simclr
```

### Full ImageNet training

Run long ImageNet jobs by pointing at the dataset root and selecting
your preferred recipe. The commands below illustrate the canonical
configurations:

```bash
# SimCLR w/ ResNet-50 and SGD
python train.py --method simclr --model resnet50 --optimizer sgd \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 256 --epochs 200 --warmup 10 --lr 0.5

# MoCo v2 w/ ResNet-34 and LARS
python train.py --method moco --model resnet34 --optimizer lars \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 256 --epochs 200 --lr 0.3 --momentum 0.9 \
  --moco-queue 65536 --knn

# BYOL w/ ResNet-18 and AdamW
python train.py --method byol --model resnet18 --optimizer adamw \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 128 --epochs 200 --lr 0.002 --weight-decay 1e-4

# SimSiam w/ ResNet-50
python train.py --method simsiam --model resnet50 --optimizer sgd \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 128 --epochs 200 --lr 0.05 --cosine

# SwAV with multi-crop augmentation
python train.py --method swav --augment swav --model resnet18 \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 256 --epochs 200 --lr 0.6 --swav-prototypes 3000 \
  --local-crops 6 --local-size 96

# Barlow Twins
python train.py --method barlow --model resnet18 --optimizer adamw \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 256 --epochs 200 --lr 0.001 --weight-decay 1e-6

# VICReg
python train.py --method vicreg --model resnet34 --optimizer sgd \
  --data imagenet --dataset-root /path/to/imagenet \
  --batch-size 256 --epochs 200 --lr 0.2 --vicreg-lambda 25 --vicreg-mu 25 --vicreg-nu 1
```

Hydra expands runs under `runs/${method}_${model}_${aug.img_size}/${now}`
by default so each experiment remains isolated.【F:configs/config.yaml†L1-L16】
Set `--run-dir` to keep checkpoints in a fixed directory. The trainer
prints smoothed loss, learning-rate, throughput (`ips`), and ETA during
training and summarizes per-epoch throughput in images/second.【F:mfcl/engines/trainer.py†L198-L288】【F:mfcl/utils/consolemonitor.py†L31-L120】

## Linear evaluation

After pretraining, attach a linear probe using `eval.py`. The
CLI mirrors `train.py` with dataset, encoder, and optimizer controls,
and accepts the checkpoint via `--checkpoint` (or `MFCL_CKPT`).【F:eval.py†L121-L219】
Example synthetic smoke test:

```bash
python eval.py --data synthetic --batch-size 32 --num-workers 0 \
  --checkpoint ./tmp_runs/simclr/ckpt_ep0001.pt \
  data.synthetic_train_size=128 data.synthetic_val_size=64 \
  linear.epochs=1 linear.lr=0.1 hydra.run.dir=./tmp_runs/linear
```

For ImageNet, simply point `--dataset-root` at the same folder used for
pretraining and adjust the probe hyperparameters if desired.

## Visualizing training curves

Convert checkpoint metrics into PDFs with `scripts/plot_metrics.py`.
Provide the run directory containing `ckpt_epXXXX.pt` files and an
optional output folder:

```bash
python scripts/plot_metrics.py --runs runs/simclr_resnet18_224/20240101_120000 --out plots/simclr
```

The script exports common metrics (`loss`, `lr`, `time_per_batch`,
`imgs_per_sec`, `knn_top1`) and gracefully skips metrics that are not
present.【F:scripts/plot_metrics.py†L1-L82】

## Smoke-test matrix

Synthetic runs that cover every method, optimizer, and augmentation
path are exercised in CI-like smoke tests. Reproduce them locally by
executing the commands below (each runs for one epoch on FakeData):

```bash
python train.py --method simclr --model resnet18 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.01 --num-workers 0 --knn --knn-period 1 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 hydra.run.dir=./tmp_runs/simclr

python train.py --method moco --model resnet34 --data synthetic --optimizer lars \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.6 --num-workers 0 --moco-queue 2048 \
  data.synthetic_train_size=128 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/moco

python train.py --method byol --model resnet18 --data synthetic --optimizer adamw \
  --epochs 1 --warmup 0 --batch-size 6 --lr 0.002 --num-workers 0 \
  data.synthetic_train_size=96 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/byol

python train.py --method simsiam --model resnet50 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 4 --lr 0.05 --num-workers 0 --img-size 224 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 hydra.run.dir=./tmp_runs/simsiam

python train.py --method barlow --model resnet18 --data synthetic --optimizer adamw \
  --epochs 1 --warmup 0 --batch-size 16 --lr 0.001 --num-workers 0 --img-size 196 \
  data.synthetic_train_size=128 data.synthetic_val_size=64 hydra.run.dir=./tmp_runs/barlow

python train.py --method vicreg --model resnet34 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 12 --lr 0.2 --num-workers 0 --vicreg-lambda 15 \
  data.synthetic_train_size=96 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/vicreg

python train.py --method swav --model resnet18 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.3 --num-workers 0 --augment swav \
  --swav-prototypes 128 --local-crops 4 --local-size 96 \
  data.synthetic_train_size=128 data.synthetic_val_size=64 hydra.run.dir=./tmp_runs/swav
```

Every command above finishes with a throughput-aware summary and
produces checkpoints compatible with the linear probe and plotting
utilities.
