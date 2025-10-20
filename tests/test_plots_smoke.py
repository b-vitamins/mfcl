import csv
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _run_plot(module: str, csv_path: Path, output_dir: Path, prefix: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            module,
            "--input",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            prefix,
        ]
    )


def test_plot_scripts_render(tmp_path):
    cases = [
        (
            "tools.plot_runtime_vs_batch",
            [
                "batch_size",
                "t_data_ms",
                "t_fwd_ms",
                "t_bwd_ms",
                "t_opt_ms",
                "t_comm_ms",
                "comm_bytes",
            ],
            [
                [64, 5.0, 12.0, 14.0, 3.5, 2.5, 4.5e7],
                [128, 6.0, 15.0, 18.0, 4.0, 3.0, 6.0e7],
            ],
            "runtime_vs_batch",
        ),
        (
            "tools.plot_accuracy_vs_time",
            ["minutes", "top1", "ci95"],
            [[10, 45.0, 0.5], [20, 51.0, 0.6], [30, 55.5, 0.4]],
            "accuracy_vs_time",
        ),
        (
            "tools.plot_memory_vs_batch",
            ["batch_size", "peak_gb"],
            [[64, 12.5], [96, 14.0], [128, 15.2]],
            "memory_vs_batch",
        ),
        (
            "tools.plot_energy_per_image",
            ["method", "images_per_s", "energy_mj"],
            [["A", 480.0, 2.1], ["B", 520.0, 1.9]],
            "energy_per_image",
        ),
        (
            "tools.plot_fidelity_vs_tau",
            ["tau", "fidelity", "ci95"],
            [[0.5, 0.92, 0.01], [0.7, 0.95, 0.008], [0.9, 0.97, 0.006]],
            "fidelity_vs_tau",
        ),
    ]

    for module, header, rows, prefix in cases:
        csv_path = tmp_path / f"{prefix}.csv"
        _write_csv(csv_path, header, rows)
        out_dir = tmp_path / prefix
        out_dir.mkdir()
        _run_plot(module, csv_path, out_dir, prefix)
        assert (out_dir / f"{prefix}.pdf").exists()
        assert (out_dir / f"{prefix}.png").exists()
