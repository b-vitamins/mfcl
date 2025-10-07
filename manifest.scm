(use-modules (guix profiles))

(specifications->manifest
  '("python-pytorch-cuda"
    "python-numpy@1"
    "python-pandas"
    "python-typer"
    "python-pyyaml"
    "python-matplotlib"))
