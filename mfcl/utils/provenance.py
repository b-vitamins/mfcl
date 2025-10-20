"""Provenance helpers for reproducible experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:  # Optional dependencies: torch/torchvision/numpy may not be installed.
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional during doc builds
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

from mfcl.utils.dist import get_local_rank, get_rank, get_world_size


def _iso_now() -> str:
    """Return the current UTC time in ISO-8601 format without microseconds."""

    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _run_git_command(args: Iterable[str]) -> str:
    try:
        out = subprocess.check_output(list(args), stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""
    return out.decode("utf-8", errors="replace").strip()


def _collect_git() -> Tuple[Dict[str, Any], str]:
    info: Dict[str, Any] = {"available": False}
    diff = ""
    sha = _run_git_command(["git", "rev-parse", "HEAD"])
    if not sha:
        return info, diff
    info["available"] = True
    info["sha"] = sha
    remote = _run_git_command(["git", "remote", "get-url", "origin"])
    info["remote"] = remote or None
    status = _run_git_command(["git", "status", "--porcelain=v1"])
    info["status"] = status
    diff = _run_git_command(["git", "diff", "HEAD"])
    info["dirty"] = bool(status.strip() or diff.strip())
    return info, diff


def _collect_runtime() -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    if torch is not None:
        runtime["torch"] = getattr(torch, "__version__", None)
        runtime["cuda_available"] = bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False
        runtime["cuda_toolkit"] = getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None
        cudnn_version = None
        try:
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():  # type: ignore[attr-defined]
                cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = None
        runtime["cudnn"] = cudnn_version
        nccl = None
        try:
            if hasattr(torch.cuda, "nccl"):
                ver = torch.cuda.nccl.version()  # type: ignore[attr-defined]
                if isinstance(ver, tuple):
                    nccl = ".".join(str(v) for v in ver)
        except Exception:
            nccl = None
        runtime["nccl"] = nccl
        amp_gpu = None
        try:
            if hasattr(torch, "get_autocast_gpu_dtype"):
                amp_gpu = str(torch.get_autocast_gpu_dtype())
        except Exception:
            amp_gpu = None
        amp_cpu = None
        try:
            if hasattr(torch, "get_autocast_cpu_dtype"):
                amp_cpu = str(torch.get_autocast_cpu_dtype())
        except Exception:
            amp_cpu = None
        runtime["amp_dtype"] = {"gpu": amp_gpu, "cpu": amp_cpu}
        deterministic: Dict[str, Any] = {}
        try:
            if hasattr(torch, "are_deterministic_algorithms_enabled"):
                deterministic["torch_deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            deterministic["torch_deterministic_algorithms"] = None
        try:
            if hasattr(torch.backends, "cudnn"):
                deterministic["cudnn_deterministic"] = bool(getattr(torch.backends.cudnn, "deterministic", False))
                deterministic["cudnn_benchmark"] = bool(getattr(torch.backends.cudnn, "benchmark", False))
        except Exception:
            deterministic["cudnn_deterministic"] = None
            deterministic["cudnn_benchmark"] = None
        try:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                deterministic["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)  # type: ignore[attr-defined]
        except Exception:
            deterministic["tf32_matmul"] = None
        runtime["deterministic"] = deterministic
    else:
        runtime.update(
            {
                "torch": None,
                "cuda_available": False,
                "cuda_toolkit": None,
                "cudnn": None,
                "nccl": None,
                "amp_dtype": {"gpu": None, "cpu": None},
                "deterministic": {},
            }
        )
    try:
        import torchvision  # type: ignore

        runtime["torchvision"] = getattr(torchvision, "__version__", None)
    except Exception:
        runtime["torchvision"] = None
    return runtime


def _cpu_model() -> str | None:
    cpu = platform.processor() or platform.uname().processor
    if cpu:
        return cpu
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    return None


def _ram_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return pages * page_size
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("memtotal"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value_kb = int(parts[1])
                        return value_kb * 1024
                    except ValueError:
                        break
    return None


def _collect_hardware() -> Dict[str, Any]:
    hardware: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "world_size": get_world_size(),
        "global_rank": get_rank(),
        "local_rank": get_local_rank(),
    }
    hardware["cpu_model"] = _cpu_model()
    hardware["ram_bytes"] = _ram_bytes()
    gpu_info: List[Dict[str, Any]] = []
    driver_version = None
    interconnect = {"pcie": None, "nvlink": None}
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        for idx in range(int(count)):
            try:
                props = torch.cuda.get_device_properties(idx)
            except Exception:
                continue
            gpu_info.append(
                {
                    "index": idx,
                    "name": getattr(props, "name", None),
                    "total_memory": getattr(props, "total_memory", None),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                    "capability": f"{getattr(props, 'major', 0)}.{getattr(props, 'minor', 0)}",
                    "pci_bus_id": getattr(props, "pci_bus_id", None),
                }
            )
        try:
            raw_driver = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
            if isinstance(raw_driver, int) and raw_driver > 0:
                major = raw_driver // 1000
                minor = (raw_driver % 1000) // 10
                driver_version = f"{major}.{minor}"
        except Exception:
            driver_version = None
        interconnect["pcie"] = any(g.get("pci_bus_id") for g in gpu_info)
        interconnect["nvlink"] = None
    hardware["gpu_count"] = len(gpu_info)
    hardware["gpus"] = gpu_info
    hardware["gpu_driver"] = driver_version
    hardware["interconnect"] = interconnect
    return hardware


def _hash_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_from_file_list(path: Path, limit: int = 1000) -> List[str]:
    sample: List[str] = []
    if not path.exists():
        return sample
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    except Exception:
        return sample
    for entry in lines[:limit]:
        sample.append(_hash_string(entry))
    return sample


def _sample_dataset(root: Path, limit: int = 1000) -> List[str]:
    sample: List[str] = []
    if not root.exists():
        return sample
    remaining = limit
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            rel = Path(dirpath).joinpath(name).relative_to(root)
            sample.append(_hash_string(str(rel)))
            remaining -= 1
            if remaining <= 0:
                return sample
    return sample


def _collect_dataset(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    root_str = data_cfg.get("root") if isinstance(data_cfg, Mapping) else None
    root_path = Path(os.path.expanduser(str(root_str))) if root_str else None
    class_index_sha: str | None = None
    sample_ids: List[str] = []
    candidates: List[Path] = []
    if isinstance(data_cfg, Mapping):
        for key in ("class_index", "class_index_file", "index_file", "train_list", "classes"):
            value = data_cfg.get(key)
            if not value:
                continue
            candidate = Path(str(value))
            if not candidate.is_absolute() and root_path is not None:
                candidate = root_path / candidate
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            class_index_sha = _sha256_file(candidate)
            break
    if candidates:
        for candidate in candidates:
            if candidate.is_file():
                sample_ids = _sample_from_file_list(candidate)
                if sample_ids:
                    break
    if not sample_ids and root_path is not None:
        sample_ids = _sample_dataset(root_path)
    return {
        "name": data_cfg.get("name") if isinstance(data_cfg, Mapping) else None,
        "root": root_str,
        "class_index_sha256": class_index_sha,
        "sample_ids": sample_ids,
    }


def _collect_seeds(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {}) if isinstance(cfg, Mapping) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    python_seed = train_cfg.get("seed") if isinstance(train_cfg, Mapping) else None
    seeds: Dict[str, Any] = {
        "python": python_seed,
        "numpy": None,
        "torch_cpu": None,
        "torch_cuda": None,
    }
    if np is not None:
        try:
            state = np.random.get_state()  # type: ignore[attr-defined]
            if state and len(state) >= 2:
                seeds["numpy"] = int(state[1][0])
        except Exception:
            seeds["numpy"] = None
    if torch is not None:
        try:
            seeds["torch_cpu"] = int(torch.initial_seed())
        except Exception:
            seeds["torch_cpu"] = None
        try:
            if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                seeds["torch_cuda"] = int(torch.cuda.initial_seed())
        except Exception:
            seeds["torch_cuda"] = None
    seeds["dataloader_workers"] = {
        "policy": data_cfg.get("worker_seed_policy", "global_seed+worker_id") if isinstance(data_cfg, Mapping) else "global_seed+worker_id",
        "seed_workers": bool(data_cfg.get("seed_workers", False)) if isinstance(data_cfg, Mapping) else False,
        "base_seed": python_seed,
    }
    return seeds


def _collect_loss_meta(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, Mapping) else {}
    return {
        "covariance_mode": loss_cfg.get("covariance_mode") if isinstance(loss_cfg, Mapping) else None,
        "moment_estimator": loss_cfg.get("moment_estimator") if isinstance(loss_cfg, Mapping) else None,
        "shrinkage_lambda": loss_cfg.get("shrinkage_lambda") if isinstance(loss_cfg, Mapping) else None,
    }


def _collect_diagnostics(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    mixture_cfg = cfg.get("mixture", {}) if isinstance(cfg, Mapping) else {}
    third_cfg = cfg.get("third_moment", {}) if isinstance(cfg, Mapping) else {}
    beta_ctrl_cfg = cfg.get("beta_ctrl", {}) if isinstance(cfg, Mapping) else {}
    return {
        "mixture": {
            "enabled": bool(mixture_cfg.get("enabled", False)) if isinstance(mixture_cfg, Mapping) else False,
            "K": mixture_cfg.get("K") if isinstance(mixture_cfg, Mapping) else None,
        },
        "third_moment": {
            "enabled": bool(third_cfg.get("enabled", False)) if isinstance(third_cfg, Mapping) else False,
        },
        "beta_ctrl": {
            "enabled": bool(beta_ctrl_cfg.get("enabled", False))
            if isinstance(beta_ctrl_cfg, Mapping)
            else False,
            "target_eps": beta_ctrl_cfg.get("target_mix_inflation_eps")
            if isinstance(beta_ctrl_cfg, Mapping)
            else None,
        },
    }


def collect_provenance(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Collect a deterministic provenance snapshot for the current process."""

    git_info, _ = _collect_git()
    snapshot: Dict[str, Any] = {
        "git": git_info,
        "runtime": _collect_runtime(),
        "hardware": _collect_hardware(),
        "seeds": _collect_seeds(run_cfg or {}),
        "dataset": _collect_dataset(run_cfg or {}),
        "loss": _collect_loss_meta(run_cfg or {}),
        "diagnostics": _collect_diagnostics(run_cfg or {}),
        "run_config": run_cfg,
    }
    return snapshot


def write_provenance(path: Path, data: Dict[str, Any]) -> None:
    """Write provenance JSON and companion artifacts to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    _, latest_diff = _collect_git()
    diff_path = path.parent / "git.diff"
    diff_path.write_text(latest_diff, encoding="utf-8")
    env_lines = [f"{key}={value}" for key, value in sorted(os.environ.items())]
    env_path = path.parent / "env.txt"
    env_path.write_text("\n".join(env_lines) + ("\n" if env_lines else ""), encoding="utf-8")


def append_event(prov_dir: Path, event: Dict[str, Any]) -> None:
    """Append a time-stamped event to provenance/events.jsonl."""

    prov_dir = Path(prov_dir)
    prov_dir.mkdir(parents=True, exist_ok=True)
    event_copy: Dict[str, Any] = dict(event)
    event_copy.setdefault("time", _iso_now())
    events_path = prov_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_copy, sort_keys=True) + "\n")


def write_stable_manifest_once(prov_dir: Path, snapshot: Dict[str, Any]) -> None:
    """Write repro.json once; subsequent calls are no-ops."""

    prov_dir = Path(prov_dir)
    prov_dir.mkdir(parents=True, exist_ok=True)
    repro_path = prov_dir / "repro.json"
    if repro_path.exists():
        return
    write_provenance(repro_path, snapshot)


__all__ = [
    "collect_provenance",
    "write_provenance",
    "append_event",
    "write_stable_manifest_once",
]
