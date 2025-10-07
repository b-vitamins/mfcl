from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from mfcl.utils.io import load_pair, pick_device
from mfcl.utils.logging import RunLogger
from mfcl.data.loaders import IIDBatchSpec, iid_batches
from mfcl.exact.gradients import exact_gradients
from mfcl.approx.gradients import mf2_gradients
from mfcl.metrics.gradients import gradient_metrics


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def main(config_path: str) -> None:
    cfg = _load_yaml(config_path)

    gcfg = (
        _load_yaml("configs/global.yaml")
        if Path("configs/global.yaml").exists()
        else {}
    )
    seed = cfg.get("seed", gcfg.get("seed", 42))
    device = pick_device(cfg.get("device", gcfg.get("device", "auto")))
    dtype_str = cfg.get("dtype", gcfg.get("dtype", "float32"))
    dtype = getattr(torch, dtype_str)
    paths = gcfg.get("paths", {})
    results_dir = cfg.get("results_dir", paths.get("results_dir", "results"))

    ds = cfg.get("dataset", {})
    dataset_name = ds.get("name", "dataset")
    x_path = ds.get("x_path")
    y_path = ds.get("y_path")
    if x_path is None:
        raise ValueError("dataset.x_path must be provided.")

    grid = cfg.get("grid", {})
    N_list = list(grid.get("N", [2048]))
    tau_list = list(grid.get("tau", [0.1]))
    objective = grid.get("objective", "infonce")
    covars = _ensure_list(grid.get("covariance", ["diag"]))
    shrinkages = _ensure_list(grid.get("shrinkage", [0.0]))
    topk = int(grid.get("topk", 10))
    batches_per_condition = int(cfg.get("batches_per_condition", 20))
    chunk_size = int(cfg.get("chunk_size", 4096))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, Y = load_pair(
        x_path=x_path, y_path=y_path, device=device, mmap=True, dtype=dtype
    )
    N_total, d = X.shape

    tag = f"gradients_{dataset_name}"
    logger = RunLogger(results_dir=results_dir, tag=tag)
    logger.log_jsonl(
        {
            "kind": "start",
            "task": "gradients",
            "dataset": dataset_name,
            "N_total": N_total,
            "d": d,
            "device": str(device),
            "dtype": str(dtype),
            "seed": seed,
            "grid": {
                "N": N_list,
                "tau": tau_list,
                "objective": objective,
                "covariance": covars,
                "shrinkage": shrinkages,
            },
        }
    )

    out_path = logger.path / "gradient_metrics.csv"
    write_header = True

    for N in N_list:
        spec = IIDBatchSpec(batch_size=N, num_batches=batches_per_condition, seed=seed)
        for b_idx, (Xb, Yb) in enumerate(iid_batches(X, Y, spec)):
            for tau in tau_list:
                gX_ex, gY_ex = exact_gradients(
                    Xb, Yb, tau=tau, objective=objective, chunk_size=chunk_size
                )

                for cov in covars:
                    lam_list = shrinkages if cov == "full" else [0.0]
                    for lam in lam_list:
                        gX_mf, gY_mf = mf2_gradients(
                            Xb,
                            Yb,
                            tau=tau,
                            objective=objective,
                            covariance=cov,
                            shrinkage=float(lam),
                        )
                        metrics = gradient_metrics(
                            gX_mf, gY_mf, gX_ex, gY_ex, topk=topk
                        )
                        row = {
                            "dataset": dataset_name,
                            "objective": objective,
                            "covariance": cov,
                            "shrinkage": float(lam),
                            "N": N,
                            "tau": float(tau),
                            "batch_id": b_idx,
                        }
                        row.update(metrics)
                        with open(out_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                            if write_header:
                                writer.writeheader()
                                write_header = False
                            writer.writerow(row)

    logger.log_jsonl({"kind": "done", "task": "gradients"})
