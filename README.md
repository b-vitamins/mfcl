# mfcl

Compose experiments with Hydra configs by groups. The default stack lives in `configs/defaults.yaml`. Examples use `--config-path` and `--config-name` so each run stays self-contained and overrideable.

Quick start:

- SimCLR baseline
  - `python examples/selfsupervised/imagenet1k/resnet18_160/simclr/train.py --config-path configs --config-name experiments/simclr_r18_160`

- Ad-hoc overrides
  - `python .../train.py --config-path configs --config-name defaults method=byol optim=adamw optim.lr=0.001 data.batch_size=192`

Hydra writes runs to `runs/${method.name}_${model.encoder}_${aug.img_size}/...`. Keep configs small and composable; no mega-CLI.
Mean-field contrastive learning
