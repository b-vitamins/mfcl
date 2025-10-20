"""Binary search helper for determining OOM-safe batch sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence


TrialRunner = Callable[[MutableMapping[str, Any], int], "TrialResult"]


@dataclass
class TrialResult:
    """Outcome of a single training attempt."""

    success: bool
    throughput: float
    peak_memory_mb: float


@dataclass
class SearchSummary:
    """Details about the best configuration discovered during search."""

    best_batch_size: int
    overrides: Dict[str, Any]
    throughput: float
    peak_memory_mb: float


def _simulate_trial(config: MutableMapping[str, Any], steps: int) -> TrialResult:
    """Fallback trial runner used in unit tests.

    The behaviour can be controlled via ``MFCL_FAKE_OOM_THRESHOLD``. The optional
    ``loss.covariance_mode`` flag acts as a multiplier on the threshold, allowing
    tests to exercise the joint-axis search logic.
    """

    del steps
    threshold = int(os.environ.get("MFCL_FAKE_OOM_THRESHOLD", "0"))
    batch_size = int(config.get("train.batch_size", 0))
    mode = str(config.get("loss.covariance_mode", "diag"))
    penalty = 1.0 if mode == "diag" else 1.25
    effective_limit = int(math.floor(threshold / penalty)) if threshold else None
    if effective_limit is not None and batch_size > effective_limit:
        raise RuntimeError("CUDA out of memory")
    throughput = batch_size / penalty if penalty > 0 else batch_size
    peak_mem = batch_size * penalty
    return TrialResult(True, throughput=float(throughput), peak_memory_mb=float(peak_mem))


def _read_peak_memory(csv_path: Path) -> float:
    peak = 0.0
    if not csv_path.exists():
        return peak
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    used = float(row.get("mem_used_MB", 0.0) or 0.0)
                    reserved = float(row.get("mem_reserved_MB", 0.0) or 0.0)
                    peak = max(peak, used, reserved)
                except Exception:
                    continue
    except Exception:
        return peak
    return peak


def _read_ips(csv_path: Path) -> float:
    """Return the median ips_step from timings.csv, or 0.0 if unavailable."""

    if not csv_path.exists():
        return 0.0
    values: List[float] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = row.get("ips_step")
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    continue
    except Exception:
        return 0.0
    if not values:
        return 0.0
    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _execute_trial(config: MutableMapping[str, Any], steps: int) -> TrialResult:
    overrides = [f"{key}={value}" for key, value in config.items()]
    batch_size = int(config.get("train.batch_size", 0))
    env = os.environ.copy()
    env["MFCL_OOM_SEARCH_MAX_STEPS"] = str(max(1, int(steps)))
    env.setdefault("HYDRA_FULL_ERROR", "1")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="oom_search_") as run_dir:
        run_path = Path(run_dir)
        cmd = [
            sys.executable,
            "train.py",
            *overrides,
            "hydra.run.dir=" + run_dir,
            "runtime.memory.enabled=true",
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")
            if "out of memory" in output.lower():
                raise RuntimeError("CUDA out of memory") from exc
            raise
        duration = max(time.perf_counter() - start, 1e-6)
        timings_csv = run_path / "timings.csv"
        ips = _read_ips(timings_csv)
        if ips <= 0.0:
            ips = (batch_size * steps) / duration if duration > 0 else 0.0
        peak_mem = _read_peak_memory(run_path / "memory.csv")
        return TrialResult(True, throughput=float(ips), peak_memory_mb=float(peak_mem))


def _enumerate_axes(axes: Mapping[str, Sequence[Any]] | None) -> List[Dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [list(axes[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for combo in product(*values):
        overrides = {keys[idx]: combo[idx] for idx in range(len(keys))}
        combos.append(overrides)
    return combos


def _merge_overrides(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(updates)
    return merged


def binary_search(
    *,
    min_batch: int,
    max_batch: int,
    steps: int,
    trial_runner: TrialRunner,
    base_overrides: Mapping[str, Any] | None = None,
    axes: Mapping[str, Sequence[Any]] | None = None,
) -> SearchSummary | None:
    """Perform a binary search for the largest batch that avoids OOM.

    Args:
        min_batch: Lower bound on the search range.
        max_batch: Upper bound on the search range.
        steps: Number of training steps each trial should survive.
        trial_runner: Callable executing a single trial.
        base_overrides: Additional overrides applied to all trials.
        axes: Optional joint search axes (cartesian product is evaluated).
    """

    if min_batch <= 0 or max_batch <= 0:
        raise ValueError("Batch size bounds must be positive integers")
    if min_batch > max_batch:
        raise ValueError("min_batch cannot exceed max_batch")

    best: SearchSummary | None = None
    base = dict(base_overrides or {})
    for axis_overrides in _enumerate_axes(axes):
        low, high = int(min_batch), int(max_batch)
        axis_best: SearchSummary | None = None
        while low <= high:
            mid = (low + high) // 2
            overrides = _merge_overrides(base, axis_overrides)
            overrides["train.batch_size"] = mid
            try:
                result = trial_runner(overrides, steps)
                success = bool(result.success)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    success = False
                    result = TrialResult(False, throughput=0.0, peak_memory_mb=0.0)
                else:
                    raise
            if success:
                axis_best = SearchSummary(
                    best_batch_size=mid,
                    overrides=dict(overrides),
                    throughput=float(result.throughput),
                    peak_memory_mb=float(result.peak_memory_mb),
                )
                low = mid + 1
            else:
                high = mid - 1
        if axis_best is None:
            continue
        if best is None or axis_best.best_batch_size > best.best_batch_size:
            best = axis_best
        elif axis_best.best_batch_size == best.best_batch_size:
            # Tie-break on throughput to prefer faster configurations.
            if axis_best.throughput > best.throughput:
                best = axis_best
    return best


def write_summary_files(summary: SearchSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "oom_search_summary.json"
    override_path = output_dir / "oom_override.yaml"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best": summary.overrides,
                "throughput": summary.throughput,
                "peak_memory_mb": summary.peak_memory_mb,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    train_bs = summary.overrides.get("train.batch_size", summary.best_batch_size)
    loss_mode = summary.overrides.get("loss.covariance_mode")
    mixture_k = summary.overrides.get("mixture.K")
    mixture_topr = summary.overrides.get("mixture.topR")

    with override_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by tools/oom_search.py\n")
        handle.write("train:\n")
        handle.write(f"  batch_size: {int(train_bs)}\n")
        if loss_mode is not None:
            handle.write("loss:\n")
            handle.write(f"  covariance_mode: {loss_mode}\n")
        if mixture_k is not None or mixture_topr is not None:
            handle.write("mixture:\n")
            if mixture_k is not None:
                handle.write(f"  K: {mixture_k}\n")
            if mixture_topr is not None:
                handle.write(f"  topR: {mixture_topr}\n")


def run_search(
    *,
    min_batch: int,
    max_batch: int,
    steps: int,
    trial_runner: TrialRunner | None = None,
    base_overrides: Mapping[str, Any] | None = None,
    axes: Mapping[str, Sequence[Any]] | None = None,
    output_dir: Path | None = None,
) -> SearchSummary:
    if trial_runner is not None:
        runner = trial_runner
    elif "MFCL_FAKE_OOM_THRESHOLD" in os.environ:
        runner = _simulate_trial
    else:
        runner = _execute_trial
    result = binary_search(
        min_batch=min_batch,
        max_batch=max_batch,
        steps=steps,
        trial_runner=runner,
        base_overrides=base_overrides,
        axes=axes,
    )
    if result is None:
        raise RuntimeError("No stable configuration found within the provided bounds")
    out_dir = output_dir or Path.cwd()
    write_summary_files(result, out_dir)
    return result


def _parse_axes(args: argparse.Namespace) -> Dict[str, Sequence[Any]]:
    axes: Dict[str, Sequence[Any]] = {}
    if args.search_covariance:
        axes["loss.covariance_mode"] = ["diag", "full"]
    if args.search_mixture and args.mixture_values:
        pairs: List[tuple[Any, Any]] = []
        for token in args.mixture_values:
            if "," in token:
                parts = token.split(",", 1)
                pairs.append((int(parts[0]), int(parts[1])))
            else:
                pairs.append((int(token), None))
        ks = sorted({pair[0] for pair in pairs})
        toprs = sorted({pair[1] for pair in pairs if pair[1] is not None})
        axes["mixture.K"] = ks
        if toprs:
            axes["mixture.topR"] = toprs
    return axes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-batch", type=int, required=True)
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path.cwd())
    parser.add_argument("--search-covariance", action="store_true")
    parser.add_argument("--search-mixture", action="store_true")
    parser.add_argument(
        "--mixture-values",
        nargs="*",
        default=(),
        help="Optional list of K or K,topR pairs used when --search-mixture is set",
    )
    args = parser.parse_args(argv)

    axes = _parse_axes(args)
    try:
        run_search(
            min_batch=args.min_batch,
            max_batch=args.max_batch,
            steps=args.steps,
            axes=axes,
            output_dir=args.output,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"oom_search failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
