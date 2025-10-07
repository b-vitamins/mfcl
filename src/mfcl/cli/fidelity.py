from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from mfcl.utils.io import load_pair, pick_device
from mfcl.utils.logging import RunLogger
from mfcl.data.loaders import IIDBatchSpec, iid_batches
from mfcl.exact.logZ import exact_loob_logmass_centered, exact_infonce_logmass
from mfcl.approx.mf2 import mf2_loob_centered, mf2_infonce_logmass
from mfcl.approx.moments import compute_neg_stats
from mfcl.approx.hybrid import hybrid_loob_centered, hybrid_infonce_logmass
from mfcl.metrics.fidelity import (
    normalization_summary_row,
    error_summary_row,
    default_tost_epsilon,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str) -> torch.device:
    return pick_device(device_str)


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def main(config_path: str) -> None:
    cfg = _load_yaml(config_path)

    # Global defaults
    gcfg = (
        _load_yaml("configs/global.yaml")
        if Path("configs/global.yaml").exists()
        else {}
    )
    seed = cfg.get("seed", gcfg.get("seed", 42))
    device = _resolve_device(cfg.get("device", gcfg.get("device", "auto")))
    dtype_str = cfg.get("dtype", gcfg.get("dtype", "float32"))
    dtype = getattr(torch, dtype_str)
    paths = gcfg.get("paths", {})
    results_dir = cfg.get("results_dir", paths.get("results_dir", "results"))

    # Dataset
    ds = cfg.get("dataset", {})
    dataset_name = ds.get("name", "dataset")
    x_path = ds.get("x_path")
    y_path = ds.get("y_path")
    if x_path is None:
        raise ValueError("dataset.x_path must be provided.")

    # Eval grid
    grid = cfg.get("grid", {})
    N_list = list(grid.get("N", [2048]))
    tau_list = list(grid.get("tau", [0.1]))
    objective = grid.get("objective", "loob")  # 'loob' or 'infonce'
    covars = _ensure_list(grid.get("covariance", ["diag"]))
    shrinkages = _ensure_list(grid.get("shrinkage", [0.0]))
    hybrid_ks = _ensure_list(grid.get("hybrid_k", []))
    batches_per_condition = int(cfg.get("batches_per_condition", 50))
    chunk_size = int(cfg.get("chunk_size", 4096))

    # Hybrid trigger
    hybrid_cfg = cfg.get("hybrid", {})
    gap_trigger: Optional[float] = hybrid_cfg.get("gap_trigger", None)

    # TOST / CI params
    tost_cfg = cfg.get("tost", {})
    n_boot = int(tost_cfg.get("n_boot", 200))
    alpha = float(tost_cfg.get("alpha", 0.10))
    eps_table = {float(k): float(v) for k, v in tost_cfg.get("eps_by_tau", {}).items()}

    # Repro
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load embeddings
    X, Y = load_pair(
        x_path=x_path, y_path=y_path, device=device, mmap=True, dtype=dtype
    )
    N_total, d = X.shape

    # Logger
    tag = f"fidelity_{dataset_name}"
    logger = RunLogger(results_dir=results_dir, tag=tag)
    logger.log_jsonl(
        {
            "kind": "start",
            "task": "fidelity",
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
                "hybrid_k": hybrid_ks,
            },
            "hybrid": {"gap_trigger": gap_trigger},
            "tost": {"n_boot": n_boot, "alpha": alpha, "eps_by_tau": eps_table},
        }
    )

    for N in N_list:
        spec = IIDBatchSpec(batch_size=N, num_batches=batches_per_condition, seed=seed)
        batch_iter = iid_batches(X, Y, spec)
        for b_idx, (Xb, Yb) in enumerate(batch_iter):
            for tau in tau_list:
                # Exact baseline per condition
                if objective == "loob":
                    exact = exact_loob_logmass_centered(
                        Xb, Yb, tau=tau, chunk_size=chunk_size
                    )
                elif objective == "infonce":
                    exact = exact_infonce_logmass(
                        Xb, Yb, tau=tau, chunk_size=chunk_size
                    )
                else:
                    raise ValueError(f"Unknown objective: {objective}")

                row_exact = normalization_summary_row(
                    exact,
                    dataset=dataset_name,
                    objective=objective,
                    N=N,
                    tau=float(tau),
                    batch_id=b_idx,
                    method="exact",
                )
                logger.append_csv("fidelity_summary.csv", row_exact)

                # Cache stats once per covariance/shrinkage pair (independent of tau)
                stats_cache = {}
                for cov in covars:
                    if cov == "full":
                        for lam in shrinkages:
                            key = (cov, float(lam))
                            stats_cache[key] = compute_neg_stats(
                                Yb, variant="full", shrinkage=float(lam)
                            )
                    else:
                        key = (cov, 0.0)
                        stats_cache[key] = compute_neg_stats(
                            Yb, variant=cov, shrinkage=0.0
                        )

                # MF2 errors
                for cov in covars:
                    lam_list = shrinkages if cov == "full" else [0.0]
                    for lam in lam_list:
                        stats = stats_cache[(cov, float(lam))]
                        if objective == "loob":
                            approx = mf2_loob_centered(
                                Xb,
                                Yb,
                                tau=tau,
                                stats=stats,
                                covariance=cov,
                                shrinkage=float(lam),
                            )
                        else:
                            approx = mf2_infonce_logmass(
                                Xb,
                                Yb,
                                tau=tau,
                                stats=stats,
                                covariance=cov,
                                shrinkage=float(lam),
                            )
                        eps = eps_table.get(
                            float(tau), default_tost_epsilon(float(tau))
                        )
                        row_err = error_summary_row(
                            approx,
                            exact,
                            dataset=dataset_name,
                            objective=objective,
                            method="mf2",
                            covariance=cov,
                            shrinkage=float(lam),
                            N=N,
                            tau=float(tau),
                            batch_id=b_idx,
                            tost_eps=float(eps),
                            n_boot=n_boot,
                            alpha=alpha,
                        )
                        logger.append_csv("fidelity_errors.csv", row_err)

                # Hybrid errors if requested
                if len(hybrid_ks) > 0:
                    for cov in covars:
                        lam_list = shrinkages if cov == "full" else [0.0]
                        for lam in lam_list:
                            stats = stats_cache[(cov, float(lam))]
                            for k in hybrid_ks:
                                k = int(k)
                                if k <= 0:
                                    continue
                                if objective == "loob":
                                    approx_h, gap = hybrid_loob_centered(
                                        Xb,
                                        Yb,
                                        tau=tau,
                                        k=k,
                                        stats=stats,
                                        covariance=cov,
                                        shrinkage=float(lam),
                                        ridge=1e-5,
                                        chunk_size=chunk_size,
                                        gap_trigger=gap_trigger,
                                    )
                                else:
                                    approx_h, gap = hybrid_infonce_logmass(
                                        Xb,
                                        Yb,
                                        tau=tau,
                                        k=k,
                                        stats=stats,
                                        covariance=cov,
                                        shrinkage=float(lam),
                                        ridge=1e-5,
                                        chunk_size=chunk_size,
                                        gap_trigger=gap_trigger,
                                    )
                                eps = eps_table.get(
                                    float(tau), default_tost_epsilon(float(tau))
                                )
                                row_h = error_summary_row(
                                    approx_h,
                                    exact,
                                    dataset=dataset_name,
                                    objective=objective,
                                    method="hybrid",
                                    covariance=cov,
                                    shrinkage=float(lam),
                                    N=N,
                                    tau=float(tau),
                                    batch_id=b_idx,
                                    tost_eps=float(eps),
                                    n_boot=n_boot,
                                    alpha=alpha,
                                    gap=gap,
                                    extra={
                                        "hybrid_k": float(k),
                                        "gap_trigger": float(gap_trigger)
                                        if gap_trigger is not None
                                        else float("nan"),
                                    },
                                )
                                logger.append_csv("fidelity_errors.csv", row_h)

    logger.log_jsonl({"kind": "done", "task": "fidelity"})
