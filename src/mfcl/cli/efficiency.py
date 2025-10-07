from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from mfcl.utils.io import load_pair
from mfcl.utils.logging import RunLogger
from mfcl.utils.timing import cuda_timing, reset_peak_memory, get_peak_memory_gb
from mfcl.utils.dist import DistEnv
from mfcl.metrics.efficiency import summarize_timing, metrics_row
from mfcl.exact.logZ import exact_infonce_logmass
from mfcl.approx.mf2 import mf2_infonce_logmass
from mfcl.approx.moments import NegStats, CovarianceKind


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


@torch.no_grad()
def _shard_rows_by_rank(T: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    N = T.shape[0]
    per = (N + world - 1) // world
    start = rank * per
    end = min((rank + 1) * per, N)
    return T[start:end].contiguous()


@torch.no_grad()
def _global_moments_from_allreduce(
    Y_local: torch.Tensor,
    variant: CovarianceKind,
    shrinkage: float,
    ridge: float,
    distenv: DistEnv,
) -> NegStats:
    device = Y_local.device
    N_local, d = Y_local.shape
    count = torch.tensor([float(N_local)], device=device, dtype=torch.float32)
    sum_local = Y_local.sum(dim=0)
    distenv.all_reduce(sum_local, op="sum")
    distenv.all_reduce(count, op="sum")
    N_global = int(count.item())
    mu = sum_local / max(N_global, 1)

    Yc_local = Y_local - mu

    if variant == "iso":
        diag_local = (Yc_local * Yc_local).mean(dim=0)
        distenv.all_reduce(diag_local, op="avg")
        alpha = float(diag_local.mean().item()) + ridge
        return NegStats(variant="iso", mu=mu, alpha=alpha, ridge=ridge)

    if variant == "diag":
        diag_local = (Yc_local * Yc_local).mean(dim=0)
        distenv.all_reduce(diag_local, op="avg")
        return NegStats(variant="diag", mu=mu, diag=diag_local + ridge, ridge=ridge)

    if variant == "full":
        Sigma_local = (Yc_local.T @ Yc_local) / float(N_local)
        distenv.all_reduce(Sigma_local, op="avg")
        tr_over_d = float(torch.trace(Sigma_local).item()) / float(d)
        if shrinkage > 0.0:
            Sigma_local = (
                1.0 - shrinkage
            ) * Sigma_local + shrinkage * tr_over_d * torch.eye(
                d, device=device, dtype=Y_local.dtype
            )
        if ridge > 0.0:
            Sigma_local = Sigma_local + ridge * torch.eye(
                d, device=device, dtype=Y_local.dtype
            )
        return NegStats(variant="full", mu=mu, Sigma=Sigma_local, ridge=ridge)

    raise ValueError(f"Unknown covariance variant: {variant}")


@torch.no_grad()
def _exact_infonce_symmetric_time(
    X_local: torch.Tensor,
    Y_local: torch.Tensor,
    tau: float,
    chunk_size: int,
    distenv: DistEnv,
) -> None:
    X_global = distenv.all_gather_concat(X_local)
    Y_global = distenv.all_gather_concat(Y_local)
    _ = exact_infonce_logmass(X_local, Y_global, tau=tau, chunk_size=chunk_size)
    _ = exact_infonce_logmass(Y_local, X_global, tau=tau, chunk_size=chunk_size)


@torch.no_grad()
def _mf2_infonce_symmetric_time(
    X_local: torch.Tensor,
    Y_local: torch.Tensor,
    tau: float,
    covariance: CovarianceKind,
    shrinkage: float,
    ridge: float,
    distenv: DistEnv,
) -> None:
    stats_y = _global_moments_from_allreduce(
        Y_local, covariance, shrinkage, ridge, distenv
    )
    _ = mf2_infonce_logmass(
        X_local,
        Y_local,
        tau=tau,
        stats=stats_y,
        covariance=covariance,
        shrinkage=shrinkage,
    )
    stats_x = _global_moments_from_allreduce(
        X_local, covariance, shrinkage, ridge, distenv
    )
    _ = mf2_infonce_logmass(
        Y_local,
        X_local,
        tau=tau,
        stats=stats_x,
        covariance=covariance,
        shrinkage=shrinkage,
    )


def main(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    gcfg = (
        _load_yaml("configs/global.yaml")
        if Path("configs/global.yaml").exists()
        else {}
    )
    seed = cfg.get("seed", gcfg.get("seed", 42))
    dtype_str = cfg.get("dtype", gcfg.get("dtype", "float32"))
    dtype = getattr(torch, dtype_str)
    paths = gcfg.get("paths", {})
    results_dir = cfg.get("results_dir", paths.get("results_dir", "results"))

    distenv = DistEnv()
    distenv.init()
    device = distenv.device

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds = cfg.get("dataset", {})
    dataset_name = ds.get("name", "dataset")
    x_path = ds.get("x_path")
    y_path = ds.get("y_path")
    if x_path is None:
        raise ValueError("dataset.x_path must be provided.")

    grid = cfg.get("grid", {})
    N_list = list(grid.get("N", [2048]))
    tau_list = list(grid.get("tau", [0.10]))
    methods = _ensure_list(grid.get("methods", ["exact", "mf2"]))
    covars = _ensure_list(grid.get("covariance", ["diag"]))
    shrinkages = _ensure_list(grid.get("shrinkage", [0.0]))
    ridge = float(grid.get("ridge", 1e-5))
    chunk_size = int(cfg.get("chunk_size", 4096))
    warmup = int(cfg.get("warmup", 20))
    reps = int(cfg.get("reps", 50))

    X_full, Y_full = load_pair(
        x_path=x_path, y_path=y_path, device=device, mmap=True, dtype=dtype
    )
    N_total, d = X_full.shape

    tag = f"efficiency_{dataset_name}"
    logger = RunLogger(results_dir=results_dir, tag=tag)
    logger.log_jsonl(
        {
            "kind": "start",
            "task": "efficiency",
            "dataset": dataset_name,
            "N_total": N_total,
            "d": d,
            "device": str(device),
            "dtype": str(dtype),
            "seed": seed,
            "world_size": distenv.world_size,
            "grid": {
                "N": N_list,
                "tau": tau_list,
                "methods": methods,
                "covariance": covars,
                "shrinkage": shrinkages,
            },
            "timing": {"warmup": warmup, "reps": reps, "chunk_size": chunk_size},
        }
    )

    out_path = logger.path / "efficiency.csv"
    write_header = True

    for N_global in N_list:
        if N_global > N_total:
            raise ValueError(
                f"Requested N_global={N_global} > available rows {N_total}"
            )
        XN = X_full[:N_global]
        YN = Y_full[:N_global]

        X_local = _shard_rows_by_rank(XN, distenv.rank, distenv.world_size)
        Y_local = _shard_rows_by_rank(YN, distenv.rank, distenv.world_size)

        for tau in tau_list:
            for method in methods:
                if method not in {"exact", "mf2"}:
                    continue

                if method == "exact":
                    distenv.reset_bytes()
                    reset_peak_memory(device)
                    for _ in range(warmup):
                        _exact_infonce_symmetric_time(
                            X_local,
                            Y_local,
                            tau=float(tau),
                            chunk_size=chunk_size,
                            distenv=distenv,
                        )

                    ms_list: List[float] = []
                    for _ in range(reps):
                        distenv.reset_bytes()
                        reset_peak_memory(device)
                        with cuda_timing(device) as timer:
                            _exact_infonce_symmetric_time(
                                X_local,
                                Y_local,
                                tau=float(tau),
                                chunk_size=chunk_size,
                                distenv=distenv,
                            )
                        distenv.barrier()
                        ms_list.append(timer.elapsed_ms())

                    ms_stats = summarize_timing(ms_list)
                    bytes_dict = distenv.get_bytes().as_dict()
                    row = metrics_row(
                        run_kind="exact",
                        dataset=dataset_name,
                        world_size=distenv.world_size,
                        device=str(device),
                        dtype=str(dtype),
                        N_global=N_global,
                        d=d,
                        method="exact",
                        covariance="n/a",
                        shrinkage=0.0,
                        reps=reps,
                        warmup=warmup,
                        ms_stats=ms_stats,
                        peak_mem_gb=get_peak_memory_gb(device),
                        **bytes_dict,
                    )
                    with open(out_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            writer.writeheader()
                            write_header = False
                        writer.writerow(row)

                else:
                    for cov in covars:
                        lam_list = shrinkages if cov == "full" else [0.0]
                        for lam in lam_list:
                            distenv.reset_bytes()
                            reset_peak_memory(device)
                            for _ in range(warmup):
                                _mf2_infonce_symmetric_time(
                                    X_local,
                                    Y_local,
                                    tau=float(tau),
                                    covariance=cov,
                                    shrinkage=float(lam),
                                    ridge=ridge,
                                    distenv=distenv,
                                )

                            ms_list: List[float] = []
                            for _ in range(reps):
                                distenv.reset_bytes()
                                reset_peak_memory(device)
                                with cuda_timing(device) as timer:
                                    _mf2_infonce_symmetric_time(
                                        X_local,
                                        Y_local,
                                        tau=float(tau),
                                        covariance=cov,
                                        shrinkage=float(lam),
                                        ridge=ridge,
                                        distenv=distenv,
                                    )
                                distenv.barrier()
                                ms_list.append(timer.elapsed_ms())

                            ms_stats = summarize_timing(ms_list)
                            bytes_dict = distenv.get_bytes().as_dict()
                            row = metrics_row(
                                run_kind="mf2",
                                dataset=dataset_name,
                                world_size=distenv.world_size,
                                device=str(device),
                                dtype=str(dtype),
                                N_global=N_global,
                                d=d,
                                method="mf2",
                                covariance=cov,
                                shrinkage=float(lam),
                                reps=reps,
                                warmup=warmup,
                                ms_stats=ms_stats,
                                peak_mem_gb=get_peak_memory_gb(device),
                                **bytes_dict,
                            )
                            with open(out_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                                if write_header:
                                    writer.writeheader()
                                    write_header = False
                                writer.writerow(row)

    logger.log_jsonl(
        {"kind": "done", "task": "efficiency", "results_csv": str(out_path)}
    )
