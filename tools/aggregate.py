"""Aggregate sweep results into tabular summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


_REQUIRED_FILES = {"timings.csv", "comms.csv", "budget.json"}
_OPTIONAL_CSV = {"memory.csv", "energy.csv", "fidelity.csv"}


@dataclass
class RunSummary:
    run_id: str
    status: str
    params: dict[str, Any]
    metadata: dict[str, Any]
    world_size: int
    run_dir: Path
    errors: list[str]
    ips_mean: float | None = None
    step_time_ms: float | None = None
    bytes_per_step: float | None = None
    budget: dict[str, Any] | None = None
    accuracy_metric: str | None = None
    accuracy_value: float | None = None
    target_value: float | None = None
    time_to_target_min: float | None = None
    energy_Wh: float | None = None
    fidelity_mean: float | None = None


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Manifest payload must be a mapping")
    return payload


def _read_csv_column(path: Path, column: str) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in reader.fieldnames:
            raise KeyError(f"Column '{column}' missing in {path.name}")
        for row in reader:
            raw = row.get(column)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def _read_timings(path: Path) -> tuple[float | None, float | None]:
    ips = _read_csv_column(path, "ips_step")
    step_ms = _read_csv_column(path, "t_step_ms")
    ips_mean = mean(ips) if ips else None
    step_mean = mean(step_ms) if step_ms else None
    return ips_mean, step_mean


def _read_comms(path: Path) -> float | None:
    bytes_total = _read_csv_column(path, "bytes_total")
    if not bytes_total:
        return None
    return mean(bytes_total)


def _load_budget(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Budget file {path} must contain a mapping")
    return payload


def _find_eval_metric(run_dir: Path, metric: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for candidate in sorted(run_dir.glob("*.csv")):
        if candidate.name in _REQUIRED_FILES or candidate.name in _OPTIONAL_CSV:
            continue
        with candidate.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or metric not in reader.fieldnames:
                continue
            for row in reader:
                payload: dict[str, Any] = {}
                for key, value in row.items():
                    if value in (None, ""):
                        continue
                    try:
                        payload[key] = float(value)
                    except ValueError:
                        payload[key] = value
                matches.append(payload)
    return matches


def _compute_time_to_target(
    metric_rows: Iterable[dict[str, Any]],
    *,
    metric_name: str,
    target: float,
    avg_step_ms: float | None,
) -> float | None:
    best_time: float | None = None
    for row in metric_rows:
        raw_value = row.get(metric_name)
        if raw_value is None:
            continue
        try:
            metric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if metric_value < target:
            continue
        if "time_minutes" in row:
            candidate = float(row["time_minutes"])
        elif "elapsed_minutes" in row:
            candidate = float(row["elapsed_minutes"])
        else:
            step = row.get("step") or row.get("global_step") or row.get("epoch")
            if step is None or avg_step_ms is None:
                continue
            candidate = float(step) * (avg_step_ms / 60000.0)
        if best_time is None or candidate < best_time:
            best_time = candidate
    return best_time


def _evaluate_metric(
    run: RunSummary,
    metric_name: str,
) -> None:
    rows = _find_eval_metric(run.run_dir, metric_name)
    if not rows:
        run.errors.append(f"Metric '{metric_name}' not found in evaluation CSVs")
        return
    def _order_key(item: dict[str, Any]) -> float:
        for key in ("step", "global_step", "epoch"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(len(rows))

    rows.sort(key=_order_key)
    final_value_raw = rows[-1].get(metric_name)
    if final_value_raw is None:
        run.accuracy_value = None
    else:
        try:
            run.accuracy_value = float(final_value_raw)
        except (TypeError, ValueError):
            run.accuracy_value = None
    run.accuracy_metric = metric_name
    target_value = run.metadata.get("target_value")
    if target_value is not None:
        try:
            target = float(target_value)
        except (TypeError, ValueError):
            run.errors.append("target_value is not numeric")
        else:
            minutes = _compute_time_to_target(
                rows,
                metric_name=metric_name,
                target=target,
                avg_step_ms=run.step_time_ms,
            )
            run.time_to_target_min = minutes
            run.target_value = target


def _collect_energy(run: RunSummary) -> None:
    energy_path = run.run_dir / "energy.csv"
    if not energy_path.exists():
        return
    try:
        with energy_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "gpu_id" not in reader.fieldnames or "energy_J_cum" not in reader.fieldnames:
                return
            last_by_gpu: dict[str, float] = {}
            for row in reader:
                gpu_key = str(row.get("gpu_id"))
                try:
                    last_by_gpu[gpu_key] = float(row.get("energy_J_cum", ""))
                except (TypeError, ValueError):
                    continue
        if last_by_gpu:
            run.energy_Wh = sum(last_by_gpu.values()) / 3600.0
    except KeyError:
        return
    except Exception:  # pragma: no cover - defensive
        return


def _collect_fidelity(run: RunSummary) -> None:
    fidelity_path = run.run_dir / "fidelity.csv"
    if not fidelity_path.exists():
        return
    values = _read_csv_column(fidelity_path, "grad_cos")
    if values:
        run.fidelity_mean = mean(values)


def _summarise_run(record: dict[str, Any]) -> RunSummary:
    run_dir = Path(record.get("run_dir", ""))
    summary = RunSummary(
        run_id=str(record.get("id")),
        status=str(record.get("status", "unknown")),
        params=dict(record.get("params") or {}),
        metadata=dict(record.get("metadata") or {}),
        world_size=int(record.get("world_size", 1)),
        run_dir=run_dir,
        errors=[],
    )
    if summary.status != "completed":
        summary.errors.append(f"Run marked as {summary.status}")
        return summary

    for required in _REQUIRED_FILES:
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"{required} missing for run {summary.run_id}")

    summary.ips_mean, summary.step_time_ms = _read_timings(run_dir / "timings.csv")
    summary.bytes_per_step = _read_comms(run_dir / "comms.csv")
    summary.budget = _load_budget(run_dir / "budget.json")

    budget_totals = summary.budget.get("totals") if isinstance(summary.budget, dict) else None
    if isinstance(budget_totals, dict):
        energy = budget_totals.get("energy_Wh")
        if isinstance(energy, (int, float)):
            summary.energy_Wh = float(energy)

    metric_name = summary.metadata.get("target_metric")
    if metric_name:
        _evaluate_metric(summary, str(metric_name))

    _collect_energy(summary)
    _collect_fidelity(summary)
    return summary


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        flat[key] = value
    return flat


def _prepare_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        flat_params = _flatten_params(run.params)
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "status": run.status,
            "world_size": run.world_size,
            "ips_mean": run.ips_mean,
            "step_time_ms": run.step_time_ms,
            "bytes_per_step": run.bytes_per_step,
            "accuracy_metric": run.accuracy_metric,
            "accuracy_value": run.accuracy_value,
            "target_value": run.target_value,
            "time_to_target_min": run.time_to_target_min,
            "energy_Wh": run.energy_Wh,
            "fidelity_mean": run.fidelity_mean,
            "errors": "; ".join(run.errors) if run.errors else "",
        }
        budget = run.budget or {}
        if isinstance(budget, dict):
            row["budget_mode"] = budget.get("mode")
            limits = budget.get("limits") if isinstance(budget.get("limits"), dict) else {}
            if isinstance(limits, dict):
                for key, value in limits.items():
                    row[f"limit_{key}"] = value
            totals = budget.get("totals") if isinstance(budget.get("totals"), dict) else {}
            if isinstance(totals, dict):
                row["steps"] = totals.get("steps")
                row["time_minutes"] = totals.get("time_minutes")
                row["tokens"] = totals.get("tokens")
                row["comm_bytes"] = totals.get("comm_bytes")
        row.update({f"param_{k}": v for k, v in flat_params.items()})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "run_id",
        "status",
        "param_model",
        "param_method",
        "param_batch_size",
        "world_size",
        "ips_mean",
        "step_time_ms",
        "bytes_per_step",
        "accuracy_metric",
        "accuracy_value",
        "time_to_target_min",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join([" --- " for _ in columns]) + "|"
    lines = [header, separator]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("nan")
                else:
                    cells.append(f"{value:.4g}")
            else:
                cells.append(str(value) if value is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")


def _aggregate(args: argparse.Namespace) -> int:
    if not args.enable_aggregator and os.environ.get("MFCL_SWEEP_AGGREGATOR") != "1":
        raise RuntimeError(
            "Aggregator is disabled. Pass --enable-aggregator to acknowledge the feature flag."
        )
    root = Path(args.root or Path.cwd())
    manifest_path = Path(args.manifest) if args.manifest else root / "sweep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    runs_data = manifest.get("runs")
    if not isinstance(runs_data, list):
        raise TypeError("Manifest must contain a list of runs")

    runs: list[RunSummary] = []
    for record in runs_data:
        if not isinstance(record, dict):
            continue
        summary = _summarise_run(record)
        runs.append(summary)

    rows = _prepare_rows(runs)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    reports_dir = (root / "reports") if args.output is None else Path(args.output)
    csv_path = reports_dir / f"summary_{timestamp}.csv"
    md_path = reports_dir / "summary.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate sweep results into CSV and Markdown reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", help="Sweep root directory (defaults to CWD).")
    parser.add_argument("--manifest", help="Explicit path to sweep_manifest.json.")
    parser.add_argument(
        "--output",
        help="Override output directory for reports (default: <root>/reports).",
    )
    parser.add_argument(
        "--enable-aggregator",
        action="store_true",
        help="Acknowledge the aggregator feature flag and enable execution.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return _aggregate(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
