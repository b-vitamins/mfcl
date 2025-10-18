# mfcl

Mean-field contrastive learning (MFCL) ships a unified, Hydra-driven
configuration stack for ImageNet-style self-supervised learning. The
repository exposes two entrypoints – `train.py` for pretraining and
`eval.py` for frozen linear evaluation – together with lean
utilities for monitoring and plotting runs.

## Environment setup

```bash
python3 -m venv .venv
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
Inspect the full option set with `python3 train.py --help`. Common flags
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
python3 train.py --method simclr --data synthetic --model resnet18 \
  --batch-size 8 --epochs 1 --lr 0.01 --num-workers 0 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 \
  hydra.run.dir=./tmp_runs/simclr
```

### Full ImageNet training

Run long ImageNet jobs by pointing at the dataset root and selecting
your preferred recipe. If you are using Guix for dependency management
or running inside a Guix‑spawned shell, prepend `guix shell -m manifest.scm --`
to the commands below (e.g., `guix shell -m manifest.scm -- python3 train.py ...`).

The commands below use ResNet‑18 with 160px crops:

```bash
# SimCLR
python3 train.py --method simclr --model resnet18 --optimizer sgd \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --warmup 10 --lr 0.25 \
  --num-workers 8 --cudnn-bench

# MoCo v2
python3 train.py --method moco --model resnet18 --optimizer lars \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --lr 0.15 --momentum 0.9 \
  --moco-queue 65536 --num-workers 8 --cudnn-bench

# BYOL
python3 train.py --method byol --model resnet18 --optimizer adamw \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --lr 0.002 --weight-decay 1e-4 \
  --num-workers 8 --cudnn-bench

# SimSiam
python3 train.py --method simsiam --model resnet18 --optimizer sgd \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --lr 0.05 --cosine \
  --num-workers 8 --cudnn-bench

# SwAV (multi‑crop)
python3 train.py --method swav --augment swav --model resnet18 --optimizer sgd \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 64 --epochs 200 --lr 0.15 --swav-prototypes 3000 \
  --local-crops 2 --local-size 96 --num-workers 8 --cudnn-bench

# Barlow Twins
python3 train.py --method barlow --model resnet18 --optimizer adamw \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --lr 0.001 --weight-decay 1e-6 \
  --num-workers 8 --cudnn-bench

# VICReg
python3 train.py --method vicreg --model resnet18 --optimizer sgd \
  --data imagenet --dataset-root /home/b/.cache/torch/datasets/imagenet/ILSVRC2012_img \
  --img-size 160 --batch-size 128 --epochs 200 --lr 0.1 \
  --vicreg-lambda 25 --vicreg-mu 25 --vicreg-nu 1 --num-workers 8 --cudnn-bench
```

Hydra expands runs under `runs/${method}_${model}_${aug.img_size}/${now}`
by default so each experiment remains isolated.【F:configs/config.yaml†L1-L16】
Set `--run-dir` to keep checkpoints in a fixed directory. The trainer
prints smoothed loss, learning-rate, throughput (`ips`), and ETA during
training and summarizes per-epoch throughput in images/second.【F:mfcl/engines/trainer.py†L198-L288】【F:mfcl/utils/consolemonitor.py†L31-L120】

### Method-specific configuration

- **BYOL momentum** – `method.byol_momentum_schedule` selects between
  `const` (default) and `cosine`. When using a schedule, set
  `method.byol_tau_base`, `method.byol_tau_final`, and
  `method.byol_momentum_schedule_steps` to control the ramp; the momentum
  updater now runs after the optimizer step for clearer semantics.
- **SimCLR negatives** – `method.ntxent_mode` toggles between the default
  "paired" NT-Xent (B−1 negatives per view) and the original `2N`
  formulation. The `2N` mode doubles the logits matrix, so monitor memory on
  large batches. Enabling `method.cross_rank_negatives=true` gathers in-batch
  negatives across ranks for DDP jobs.
- **MoCo queue** – keep the per-rank FIFO (default) or turn on
  `method.cross_rank_queue=true` to gather keys before enqueuing so every
  rank sees the global queue. You can also enable SyncBatchNorm via
  `method.use_syncbn=true` before DDP wrapping.
- **SwAV assignments** – tune Sinkhorn stability with
  `method.swav_sinkhorn_tol`, `method.swav_sinkhorn_max_iters`, and
  `method.swav_use_fp32_sinkhorn`. Set `method.swav_codes_queue_size>0`
  to maintain a small FIFO of past logits for the global crops, improving
  assignment stability on small batches.

Notes

- AMP is enabled by default; the above keeps it that way for memory headroom.
- You can omit `--dataset-root` if your path matches the default.
- SwAV uses multi-crop; if it still OOMs, try `--batch-size 48` or reduce `--local-crops 1`.

## Distributed training

`torchrun` is the recommended launcher for both single-node multi-GPU and
multi-node clusters. The training entrypoint auto-detects the usual environment
variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`),
initializes `torch.distributed` with NCCL on CUDA builds (falling back to gloo
when no GPUs are present), and wraps the method with `DistributedDataParallel`
when `WORLD_SIZE > 1`. Samplers are sharded with `DistributedSampler`, their
epoch is advanced each loop, and checkpoints/logs are kept on rank 0 only.【F:train.py†L19-L142】【F:mfcl/core/factory.py†L708-L764】【F:mfcl/engines/trainer.py†L167-L306】
Validation loaders stay unsharded so rank 0 can run kNN monitors without
cross-rank feature gathering.【F:mfcl/core/factory.py†L735-L764】

### Single-node examples

```bash
# 8 GPUs on one host
torchrun --standalone --nproc_per_node=8 \
  python3 train.py --method simclr --model resnet18 --data imagenet \
  --dataset-root /datasets/imagenet --img-size 160 --batch-size 128
```

### Multi-node via torchrun

```bash
# Node 0
MASTER_ADDR=host0 MASTER_PORT=29500 NODE_RANK=0 WORLD_SIZE=16 \
  torchrun --nnodes=2 --nproc_per_node=8 \
  python3 train.py --method simclr --model resnet18 --data imagenet

# Node 1
MASTER_ADDR=host0 MASTER_PORT=29500 NODE_RANK=1 WORLD_SIZE=16 \
  torchrun --nnodes=2 --nproc_per_node=8 \
  python3 train.py --method simclr --model resnet18 --data imagenet
```

On Slurm clusters, launch one process per GPU and delegate rendezvous to
`torchrun` (e.g., `srun --ntasks-per-node=8 torchrun --nproc_per_node=8 ...`).
Set `NCCL_SOCKET_IFNAME` to your high-speed interface, and keep
`NCCL_P2P_DISABLE=0` / `NCCL_IB_DISABLE=0` to leverage NVLink or InfiniBand.
Enable `NCCL_DEBUG=INFO` for verbose debugging if rendezvous fails.

During training each rank logs its local throughput while the summary line also
reports a global images/sec aggregate (sum of samples divided by the slowest
epoch).【F:mfcl/engines/trainer.py†L198-L306】 Checkpoints (`ckpt_epXXXX.pt` and
`latest.pt`) are created by rank 0; other ranks reuse them when resuming runs.

- Pass `--ddp-find-unused-parameters` if your research branch introduces
  conditional computation that would otherwise drop gradients before the
  reducer sees them.【F:train.py†L120-L191】
- MoCo keeps a per-rank negative queue by default. Set
  `method.cross_rank_queue=true` to all-gather keys before enqueuing so the
  queue reflects the entire world size.
- Sync BatchNorm is optional; call
  `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` before DDP wrapping if
  you would like to enable it.

### Scheduler stepping modes

The `train.scheduler_step_on` flag controls whether LR schedulers advance per
optimizer update (`batch`) or per epoch (`epoch`). In batch mode warmup and
cosine schedules are constructed from the number of optimizer steps per epoch,
so `warmup_epochs` and `epochs` translate to warmup iterations and `T_max`
automatically.【F:mfcl/core/factory.py†L809-L855】【F:tests/unit/test_scheduler_build.py†L28-L52】

## Linear evaluation

After pretraining, attach a linear probe using `eval.py`. The
CLI mirrors `train.py` with dataset, encoder, and optimizer controls,
and accepts the checkpoint via `--checkpoint` (or `MFCL_CKPT`).【F:eval.py†L121-L219】
Example synthetic smoke test:

```bash
python3 eval.py --data synthetic --batch-size 32 --num-workers 0 \
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
python3 scripts/plot_metrics.py --runs runs/simclr_resnet18_224/20240101_120000 --out plots/simclr
```

The script exports common metrics (`loss`, `lr`, `time_per_batch`,
`imgs_per_sec`, `knn_top1`) and gracefully skips metrics that are not
present.【F:scripts/plot_metrics.py†L1-L82】

## Smoke-test matrix

Synthetic runs that cover every method, optimizer, and augmentation
path are exercised in CI-like smoke tests. Reproduce them locally by
executing the commands below (each runs for one epoch on FakeData):

```bash
python3 train.py --method simclr --model resnet18 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.01 --num-workers 0 --knn --knn-period 1 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 hydra.run.dir=./tmp_runs/simclr

python3 train.py --method moco --model resnet34 --data synthetic --optimizer lars \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.6 --num-workers 0 --moco-queue 2048 \
  data.synthetic_train_size=128 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/moco

python3 train.py --method byol --model resnet18 --data synthetic --optimizer adamw \
  --epochs 1 --warmup 0 --batch-size 6 --lr 0.002 --num-workers 0 \
  data.synthetic_train_size=96 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/byol

python3 train.py --method simsiam --model resnet50 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 4 --lr 0.05 --num-workers 0 --img-size 224 \
  data.synthetic_train_size=64 data.synthetic_val_size=32 hydra.run.dir=./tmp_runs/simsiam

python3 train.py --method barlow --model resnet18 --data synthetic --optimizer adamw \
  --epochs 1 --warmup 0 --batch-size 16 --lr 0.001 --num-workers 0 --img-size 196 \
  data.synthetic_train_size=128 data.synthetic_val_size=64 hydra.run.dir=./tmp_runs/barlow

python3 train.py --method vicreg --model resnet34 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 12 --lr 0.2 --num-workers 0 --vicreg-lambda 15 \
  data.synthetic_train_size=96 data.synthetic_val_size=48 hydra.run.dir=./tmp_runs/vicreg

python3 train.py --method swav --model resnet18 --data synthetic --optimizer sgd \
  --epochs 1 --warmup 0 --batch-size 8 --lr 0.3 --num-workers 0 --augment swav \
  --swav-prototypes 128 --local-crops 4 --local-size 96 \
  data.synthetic_train_size=128 data.synthetic_val_size=64 hydra.run.dir=./tmp_runs/swav
```

Every command above finishes with a throughput-aware summary and
produces checkpoints compatible with the linear probe and plotting
utilities.
