from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha
    except Exception:
        return None


class RunLogger:
    """
    Minimal run logger that writes:
      - JSONL file with config/env summary
      - CSV appenders for metrics
    Creates a timestamped run directory under results_dir.
    """

    def __init__(self, results_dir: str, tag: str) -> None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(results_dir) / f"{ts}_{tag}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "run.jsonl"

        # Create empty jsonl
        self.jsonl_path.touch(exist_ok=True)

    @property
    def path(self) -> Path:
        return self.run_dir

    def log_jsonl(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        if "git_sha" not in record:
            record["git_sha"] = _git_sha()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_csv(self, filename: str, row: Dict[str, Any]) -> None:
        csv_path = self.run_dir / filename
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
