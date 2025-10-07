# Testing & CI

## Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .
pip install pytest
pytest -q
```

Covers:

- covariance stats (iso/diag/full) inc. low-rank quadratic check
- exact InfoNCE/LOOB streaming log-sum-exp vs naive
- hybrid `k = N-1` equivalence with exact
- exact symmetric gradients vs autograd reference
- MF2 gradient direction sanity (cosine)
- timing/dist helpers on CPU single-process

## CI

GitHub Actions workflow `.github/workflows/ci.yaml` runs the test suite on Python 3.10 and 3.11 with CPU PyTorch wheels.
