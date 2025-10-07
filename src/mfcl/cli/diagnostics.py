from __future__ import annotations

import csv
import math
from typing import Any, Dict, List

import torch
import yaml

from mfcl.utils.io import load_pair, pick_device
from mfcl.utils.logging import RunLogger
from mfcl.data.loaders import IIDBatchSpec, iid_batches
from mfcl.exact.logZ import exact_loob_logmass_centered, exact_infonce_logmass
from mfcl.approx.mf2 import mf2_loob_centered, mf2_infonce_logmass
from mfcl.approx.moments import compute_neg_stats
from mfcl.metrics.spectra import spectral_diagnostics
from mfcl.metrics.fidelity import error_metrics


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def main(config_path: str) -> None:
    cfg = _load_yaml(config_path)

    seed = cfg.get("seed", 42)
    device = pick_device(cfg.get("device", "auto"))
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    results_dir = cfg.get("results_dir", "results")

    ds = cfg.get("dataset", {})
    dataset_name = ds.get("name", "dataset")
    x_path = ds.get("x_path")
    y_path = ds.get("y_path")
    if x_path is None:
        raise ValueError("dataset.x_path must be provided.")

    grid = cfg.get("grid", {})
    N_list = list(grid.get("N", [2048]))
    tau_list = list(grid.get("tau", [0.10, 0.25]))
    objective = grid.get("objective", "infonce")
    covariance = grid.get("covariance", "full")
    shrinkage = float(grid.get("shrinkage", 0.05))
    lowrank_r = grid.get("lowrank_r", None)
    batches_per_condition = int(cfg.get("batches_per_condition", 50))
    r_list = list(cfg.get("spectral_r", [8, 16, 32]))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, Y = load_pair(
        x_path=x_path, y_path=y_path, device=device, mmap=True, dtype=dtype
    )

    logger = RunLogger(results_dir=results_dir, tag=f"diagnostics_{dataset_name}")
    diag_csv = logger.path / "diagnostics.csv"
    pos_csv = logger.path / "pos_reentry_stats.csv"
    write_diag = True
    write_pos = True

    for N in N_list:
        spec = IIDBatchSpec(batch_size=N, num_batches=batches_per_condition, seed=seed)
        for batch_id, (Xb, Yb) in enumerate(iid_batches(X, Y, spec)):
            spec_stats = spectral_diagnostics(Yb, r_list=r_list)

            for tau in tau_list:
                beta = 1.0 / float(tau)
                if objective == "loob":
                    exact = exact_loob_logmass_centered(Xb, Yb, tau=tau)
                    stats = compute_neg_stats(
                        Yb,
                        variant=covariance,
                        shrinkage=shrinkage,
                        lowrank_r=lowrank_r,
                    )
                    approx = mf2_loob_centered(
                        Xb,
                        Yb,
                        tau=tau,
                        stats=stats,
                        covariance=covariance,
                        shrinkage=shrinkage,
                    )
                else:
                    exact = exact_infonce_logmass(Xb, Yb, tau=tau)
                    stats = compute_neg_stats(
                        Yb,
                        variant=covariance,
                        shrinkage=shrinkage,
                        lowrank_r=lowrank_r,
                    )
                    approx = mf2_infonce_logmass(
                        Xb,
                        Yb,
                        tau=tau,
                        stats=stats,
                        covariance=covariance,
                        shrinkage=shrinkage,
                    )

                em = error_metrics(approx, exact)
                row = {
                    "dataset": dataset_name,
                    "objective": objective,
                    "covariance": covariance,
                    "shrinkage": shrinkage,
                    "lowrank_r": -1 if lowrank_r is None else int(lowrank_r),
                    "N": N,
                    "tau": float(tau),
                    "batch_id": batch_id,
                    **spec_stats,
                    **em,
                }
                with open(diag_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_diag:
                        writer.writeheader()
                        write_diag = False
                    writer.writerow(row)

                if objective == "infonce":
                    mu_proj = Xb @ stats.mu
                    quad = stats.quad(Xb)
                    m_neg = (
                        math.log(max(N - 1, 1))
                        + beta * mu_proj
                        + 0.5 * (beta**2) * quad
                    )
                    a_pos = beta * (Xb * Yb).sum(dim=1)
                    delta = torch.nn.functional.softplus(a_pos - m_neg)

                    prow = {
                        "dataset": dataset_name,
                        "N": N,
                        "tau": float(tau),
                        "batch_id": batch_id,
                        "delta_q25": float(delta.quantile(0.25).item()),
                        "delta_median": float(delta.quantile(0.5).item()),
                        "delta_q75": float(delta.quantile(0.75).item()),
                        "frac_pos_dominates": float(
                            (a_pos - m_neg > 0).float().mean().item()
                        ),
                        "covariance": covariance,
                        "shrinkage": shrinkage,
                        "lowrank_r": -1 if lowrank_r is None else int(lowrank_r),
                    }
                    with open(pos_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(prow.keys()))
                        if write_pos:
                            writer.writeheader()
                            write_pos = False
                        writer.writerow(prow)

    logger.log_jsonl(
        {
            "kind": "done",
            "task": "diagnostics",
            "diag_csv": str(diag_csv),
            "pos_csv": str(pos_csv),
        }
    )
