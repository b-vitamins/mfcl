"""Grid runner for distributed MFCL sweeps.

This module expands a YAML grid description into individual runs launched via
``torchrun``. Each run records its configuration, output directory, and final
status inside a manifest that downstream tooling (e.g. :mod:`tools.aggregate`)
consumes. All behaviour is gated behind an explicit feature flag to comply with
the repository guardrails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import OmegaConf


# group selectors are represented via the special @group:<name> marker so we can
# emit Hydra overrides without JSON quoting (e.g. model=resnet50)
_PARAM_ALIASES = {
    "model": "@group:model",
    "method": "@group:method",
    "batch_size": "data.batch_size",
    "tau": "method.tau",
}

_SPECIAL_KEYS = {"world_size"}

_BUDGET_LIMIT_KEYS = {
    "iso_time": "max_minutes",
    "iso_tokens": "max_tokens",
    "iso_epochs": "max_epochs",
    "comm_cap": "max_comm_bytes",
    "energy_cap": "max_energy_Wh",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_dict(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    try:
        return dict(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("Grid configuration must be convertible to a dict") from exc


def _expand_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    if not params:
        return [{}]
    keys: list[str] = []
    axes: list[Sequence[Any]] = []
    for key, value in params.items():
        keys.append(str(key))
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Parameter '{key}' has an empty sweep axis")
            axes.append(list(value))
        else:
            axes.append([value])
    combos: list[dict[str, Any]] = []
    for values in product(*axes):
        combo = {keys[i]: values[i] for i in range(len(keys))}
        combos.append(combo)
    return combos


def _format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif value is None:
        literal = "null"
    elif isinstance(value, (int, float)):
        literal = str(value)
    elif isinstance(value, str):
        literal = json.dumps(value)
    else:
        literal = json.dumps(value)
    return f"{key}={literal}"


def _sanitize_slug(name: str) -> str:
    safe = name.strip().replace(" ", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"-", "_"})
    return safe or "run"


def _normalize_budget(budget_cfg: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
    if not budget_cfg:
        return [], {}
    config = _as_dict(budget_cfg)
    mode = str(config.get("mode", ""))
    if not mode:
        raise ValueError("Budget configuration must include a 'mode'")
    mode = mode.lower()
    if mode not in _BUDGET_LIMIT_KEYS:
        raise ValueError(f"Unsupported budget mode: {mode}")
    limit_key = _BUDGET_LIMIT_KEYS[mode]
    limit_value = config.get("limit", config.get(limit_key))
    if limit_value is None:
        raise ValueError(f"Budget mode '{mode}' requires '{limit_key}' or 'limit'")

    overrides = [
        "runtime.budget.enabled=true",
        _format_override("runtime.budget.mode", mode),
        _format_override(f"runtime.budget.{limit_key}", limit_value),
    ]

    for optional_key in ("max_comm_bytes", "max_energy_Wh", "steps_per_epoch"):
        if optional_key in config and config[optional_key] is not None:
            overrides.append(
                _format_override(f"runtime.budget.{optional_key}", config[optional_key])
            )

    metadata = {
        "budget_mode": mode,
        "budget_limit_key": limit_key,
        "budget_limit_value": limit_value,
    }
    for key in ("target_metric", "target_value"):
        if key in config:
            metadata[key] = config[key]
    return overrides, metadata


def _resolve_param(key: str, value: Any) -> tuple[str | None, str | None, Any]:
    if key in _SPECIAL_KEYS:
        return key, None, value
    alias = _PARAM_ALIASES.get(key, key)
    return key, alias, value


def _coerce_nnodes(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise ValueError("nnodes must be a positive integer")
        return str(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    return value_str


def _coerce_node_rank(value: Any) -> int | None:
    if value is None:
        return None
    try:
        node_rank = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("node_rank must be an integer") from exc
    if node_rank < 0:
        raise ValueError("node_rank must be non-negative")
    return node_rank


def _normalize_rendezvous(cfg: Any) -> dict[str, str]:
    data: dict[str, str] = {}
    if not cfg:
        return data
    for key, value in _as_dict(cfg).items():
        if value is None:
            continue
        data[str(key)] = str(value)
    return data


@dataclass
class RunSpec:
    identifier: str
    run_dir: Path
    world_size: int
    overrides: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    nnodes: str | None = None
    node_rank: int | None = None
    rendezvous: dict[str, str] = field(default_factory=dict)
    status: str = "pending"
    returncode: int | None = None
    command: list[str] = field(default_factory=list)
    launcher_log: Path | None = None
    started_at: str | None = None
    finished_at: str | None = None


def _build_run_specs(
    *,
    grid_cfg: dict[str, Any],
    output_root: Path,
    base_overrides: Iterable[str],
    base_env: dict[str, str],
) -> tuple[list[RunSpec], dict[str, str]]:
    grid_entries = grid_cfg.get("grid")
    if not isinstance(grid_entries, list) or not grid_entries:
        raise ValueError("Grid configuration must include a non-empty 'grid' list")

    default_nnodes = _coerce_nnodes(grid_cfg.get("nnodes"))
    default_node_rank = _coerce_node_rank(grid_cfg.get("node_rank"))
    default_rendezvous = _normalize_rendezvous(grid_cfg.get("rendezvous"))

    runs: list[RunSpec] = []
    counter = 0
    for entry_index, raw_entry in enumerate(grid_entries):
        entry = _as_dict(raw_entry)
        entry_name = entry.get("name") or f"group{entry_index:02d}"
        entry_slug = _sanitize_slug(str(entry_name))
        entry_overrides = list(base_overrides)
        for override in entry.get("overrides", []) or []:
            entry_overrides.append(str(override))
        entry_env: dict[str, str] = {}
        if "env" in entry:
            entry_env = {str(k): str(v) for k, v in _as_dict(entry["env"]).items()}
        params_dict = _as_dict(entry.get("params"))
        combos = _expand_params(params_dict)
        budget_overrides, budget_meta = _normalize_budget(entry.get("budget"))
        entry_overrides.extend(budget_overrides)
        entry_metadata = _as_dict(entry.get("metadata"))
        entry_metadata.update(budget_meta)

        entry_nnodes = _coerce_nnodes(entry.get("nnodes") if "nnodes" in entry else default_nnodes)
        entry_node_rank = _coerce_node_rank(entry.get("node_rank") if "node_rank" in entry else default_node_rank)
        rendezvous_cfg = dict(default_rendezvous)
        if "rendezvous" in entry:
            rendezvous_cfg.update(_normalize_rendezvous(entry.get("rendezvous")))

        for combo_index, combo in enumerate(combos):
            counter += 1
            combo_params: dict[str, Any] = {}
            overrides: list[str] = list(entry_overrides)
            combo_env: dict[str, str] = {}
            world_size = 1
            for raw_key, raw_value in combo.items():
                key_name, alias, value = _resolve_param(str(raw_key), raw_value)
                combo_params[key_name] = value
                if key_name == "world_size":
                    try:
                        world_size = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("world_size must be an integer") from exc
                    continue
                if isinstance(alias, str) and alias.startswith("@group:"):
                    group_name = alias.split(":", 1)[1]
                    overrides.append(f"{group_name}={value}")
                    continue
                if key_name == "amp_dtype":
                    dtype = str(value).lower()
                    if dtype not in {"fp16", "bf16", "fp32"}:
                        raise ValueError("amp_dtype must be one of: fp16, bf16, fp32")
                    overrides.append(
                        "train.loss_fp32=true" if dtype == "fp32" else "train.loss_fp32=false"
                    )
                    combo_env["MFCL_AMP_DTYPE"] = dtype
                    continue
                if alias is not None:
                    overrides.append(_format_override(alias, value))

            run_id = f"{counter:03d}_{entry_slug}"
            if len(combos) > 1:
                run_id = f"{run_id}_{combo_index:02d}"
            run_dir = output_root / run_id
            overrides.append(_format_override("train.save_dir", str(run_dir)))
            overrides.append(_format_override("hydra.run.dir", str(run_dir)))
            overrides.append("hydra.output_subdir=null")
            overrides.append("hydra.job.chdir=false")

            spec = RunSpec(
                identifier=run_id,
                run_dir=run_dir,
                world_size=max(1, world_size),
                overrides=overrides,
                params=combo_params,
                metadata=dict(entry_metadata),
                env={**entry_env, **combo_env},
                nnodes=entry_nnodes,
                node_rank=entry_node_rank,
                rendezvous=rendezvous_cfg,
            )
            runs.append(spec)
    return runs, {str(k): str(v) for k, v in base_env.items()}


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _launch_run(
    spec: RunSpec,
    *,
    entrypoint: str,
    base_cmd: list[str],
    base_env: dict[str, str],
    working_dir: Path,
) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    command = list(base_cmd)
    nnodes_flag_added = False
    if spec.nnodes:
        command.append(f"--nnodes={spec.nnodes}")
        nnodes_flag_added = True
    if spec.node_rank is not None:
        command.append(f"--node_rank={spec.node_rank}")

    rendezvous_flags = {
        "backend": "rdzv_backend",
        "endpoint": "rdzv_endpoint",
        "id": "rdzv_id",
        "conf": "rdzv_conf",
    }
    rendezvous_present = bool(spec.rendezvous)
    for key, flag in rendezvous_flags.items():
        value = spec.rendezvous.get(key)
        if value:
            command.append(f"--{flag}={value}")

    if not rendezvous_present and not nnodes_flag_added and spec.node_rank is None:
        command.append("--standalone")

    command.append(f"--nproc_per_node={spec.world_size}")
    command.append(entrypoint)
    command.extend(spec.overrides)
    spec.command = command

    child_env = os.environ.copy()
    child_env.update(base_env)
    child_env.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in child_env:
        import socket

        with socket.socket() as sock:
            sock.bind(("", 0))
            child_env["MASTER_PORT"] = str(sock.getsockname()[1])
    try:
        nnodes_int = int(spec.nnodes) if spec.nnodes is not None else None
    except ValueError:
        nnodes_int = None
    total_world_size = spec.world_size * nnodes_int if nnodes_int else spec.world_size
    child_env["WORLD_SIZE"] = str(total_world_size)
    child_env["LOCAL_WORLD_SIZE"] = str(spec.world_size)
    child_env["NPROC_PER_NODE"] = str(spec.world_size)
    if spec.nnodes:
        child_env["NNODES"] = str(spec.nnodes)
    if spec.node_rank is not None:
        child_env["NODE_RANK"] = str(spec.node_rank)
    child_env["MFCL_SWEEP_RUN_ID"] = spec.identifier
    child_env["MFCL_SWEEP_RUN_DIR"] = str(spec.run_dir)
    child_env["MFCL_SWEEP_PARAMS"] = json.dumps(spec.params)
    child_env["MFCL_SWEEP_METADATA"] = json.dumps(spec.metadata)
    child_env.update(spec.env)
    child_env["HYDRA_FULL_ERROR"] = "1"

    log_path = spec.run_dir / "sweep.log"
    spec.launcher_log = log_path

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            spec.started_at = datetime.utcnow().isoformat() + "Z"
            result = subprocess.run(
                command,
                cwd=str(working_dir),
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            spec.returncode = int(result.returncode)
            if result.returncode == 0:
                spec.status = "completed"
            else:
                spec.status = "error"
        except KeyboardInterrupt:  # pragma: no cover - manual interrupt
            spec.status = "cancelled"
            spec.returncode = -1
            raise
        except Exception as exc:  # pragma: no cover - defensive
            spec.status = "error"
            spec.returncode = -1
            log_file.write(f"\n[SWEEP] Launcher raised: {exc!r}\n")
        finally:
            spec.finished_at = datetime.utcnow().isoformat() + "Z"


def _build_manifest_payload(
    *,
    name: str,
    grid_path: Path,
    output_root: Path,
    runs: Sequence[RunSpec],
) -> dict[str, Any]:
    payload = {
        "name": name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "grid_path": str(grid_path.resolve()),
        "output_dir": str(output_root),
        "runs": [],
    }
    for spec in runs:
        payload["runs"].append(
            {
                "id": spec.identifier,
                "run_dir": str(spec.run_dir),
                "world_size": spec.world_size,
                "overrides": list(spec.overrides),
                "params": dict(spec.params),
                "metadata": dict(spec.metadata),
                "env": dict(spec.env),
                "nnodes": spec.nnodes,
                "node_rank": spec.node_rank,
                "rendezvous": dict(spec.rendezvous),
                "status": spec.status,
                "returncode": spec.returncode,
                "command": list(spec.command),
                "launcher_log": str(spec.launcher_log) if spec.launcher_log else None,
                "started_at": spec.started_at,
                "finished_at": spec.finished_at,
            }
        )
    return payload


def _persist_manifest(manifest_path: Path, runs: Sequence[RunSpec], template: dict[str, Any]) -> None:
    for spec, payload in zip(runs, template["runs"], strict=False):
        payload.update(
            {
                "status": spec.status,
                "returncode": spec.returncode,
                "command": list(spec.command),
                "launcher_log": str(spec.launcher_log) if spec.launcher_log else None,
                "started_at": spec.started_at,
                "finished_at": spec.finished_at,
            }
        )
    _write_manifest(manifest_path, template)


def _run_sweep(args: argparse.Namespace) -> int:
    if not args.enable_sweeps and os.environ.get("MFCL_SWEEPS_ENABLED") != "1":
        raise RuntimeError(
            "Sweep runner is disabled. Pass --enable-sweeps to acknowledge the feature flag."
        )

    grid_path = Path(args.grid).resolve()
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    grid_cfg = OmegaConf.to_container(OmegaConf.load(str(grid_path)), resolve=True)  # type: ignore[arg-type]
    if not isinstance(grid_cfg, dict):
        raise TypeError("Grid configuration must be a mapping")

    sweep_name = str(grid_cfg.get("name") or grid_path.stem)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    default_root = _project_root() / "runs" / f"{sweep_name}_{timestamp}"
    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    elif grid_cfg.get("output_dir"):
        output_root = Path(str(grid_cfg["output_dir"]))
    else:
        output_root = default_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_env = {str(k): str(v) for k, v in _as_dict(grid_cfg.get("env")).items()}
    base_overrides = [str(item) for item in grid_cfg.get("overrides", []) or []]

    runs, merged_env = _build_run_specs(
        grid_cfg=grid_cfg,
        output_root=output_root,
        base_overrides=base_overrides,
        base_env=base_env,
    )

    base_cmd = ["torchrun"]
    entrypoint = str(grid_cfg.get("entrypoint") or "train.py")
    working_dir = Path(grid_cfg.get("work_dir") or _project_root())

    manifest_path = output_root / "sweep_manifest.json"
    manifest_payload = _build_manifest_payload(
        name=sweep_name,
        grid_path=grid_path,
        output_root=output_root,
        runs=runs,
    )
    _persist_manifest(manifest_path, runs, manifest_payload)

    for spec in runs:
        payload_index = next(
            (idx for idx, item in enumerate(manifest_payload["runs"]) if item["id"] == spec.identifier),
            None,
        )
        if payload_index is None:  # pragma: no cover - defensive
            continue
        try:
            _launch_run(
                spec,
                entrypoint=entrypoint,
                base_cmd=base_cmd,
                base_env=merged_env,
                working_dir=working_dir,
            )
        finally:
            manifest_payload["runs"][payload_index].update(
                {
                    "status": spec.status,
                    "returncode": spec.returncode,
                    "command": list(spec.command),
                    "launcher_log": str(spec.launcher_log) if spec.launcher_log else None,
                    "started_at": spec.started_at,
                    "finished_at": spec.finished_at,
                }
            )
            _persist_manifest(manifest_path, runs, manifest_payload)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand a sweep grid and launch runs via torchrun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("grid", help="Path to the YAML sweep grid definition.")
    parser.add_argument(
        "--output-dir",
        help="Optional override for the sweep output directory.",
    )
    parser.add_argument(
        "--enable-sweeps",
        action="store_true",
        help="Acknowledge the sweep feature flag and enable execution.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return _run_sweep(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
