;; Guix manifest

(specifications->manifest
 '(;; Core Python
   "python"
   ;; Runtime dependencies
   "python-pytorch-cuda"
   "python-torchvision-cuda"
   "python-timm"
   "python-hydra-core"
   "python-numpy@1"
   "python-scipy"
   "python-scikit-learn"
   "python-matplotlib"
   "python-pillow"
   ;; Development and test dependencies
   "python-pytest"
   "python-pytest-cov"
   "python-hypothesis"
   "python-ruff"
   "node-pyright"))
