"""Low-rank third-moment sketch for embedding diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

from mfcl.utils import dist as dist_utils


class ThirdMomentSketch:
    """Maintain a random-feature sketch of the third central moment."""

    _CSV_COLUMNS = ["step", "epoch", "p50_abs", "p90_abs"]

    def __init__(
        self,
        *,
        rank: int = 16,
        seed: int = 41,
        ema_decay: float = 0.95,
        enabled: bool = False,
        is_main: bool = True,
        log_dir: Optional[str | Path] = None,
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in the range [0, 1)")

        self.rank = int(rank)
        self.seed = int(seed)
        self.ema_decay = float(ema_decay)
        self.enabled = bool(enabled)
        self.is_main = bool(is_main)

        self._dim: Optional[int] = None
        self._projection: Optional[torch.Tensor] = None
        self._ema_u: Optional[torch.Tensor] = None
        self._mean: Optional[torch.Tensor] = None
        self._last_summary: Optional[Dict[str, float]] = None
        self._rngs: Dict[str, torch.Generator] = {}
        cpu_device = torch.device("cpu")
        cpu_generator = torch.Generator(device=cpu_device)
        cpu_generator.manual_seed(self.seed)
        self._rngs[str(cpu_device)] = cpu_generator

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        if self.enabled and self.is_main and log_dir is not None:
            path = Path(log_dir).joinpath("kappa3.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = path.exists()
            self._csv_file = path.open("a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._CSV_COLUMNS
            )
            if (not file_exists) or path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._csv_file.flush()
            self._csv_path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def has_mean(self) -> bool:
        """Return True if a mean vector has been registered."""

        return self._mean is not None

    def set_mean(self, mean: torch.Tensor) -> None:
        """Register the reference mean ``μ`` used for centering."""

        if not torch.is_tensor(mean):
            raise TypeError("mean must be a torch.Tensor")
        if mean.ndim != 1:
            raise ValueError("mean must be a 1D tensor")
        mean_f32 = mean.detach().to(dtype=torch.float32, device="cpu")
        if self._dim is None:
            self._dim = int(mean_f32.numel())
        elif self._dim != int(mean_f32.numel()):
            raise ValueError("mean dimensionality changed")
        self._mean = mean_f32.clone()

    def update(
        self,
        embeddings: torch.Tensor,
        *,
        mean: Optional[torch.Tensor] = None,
        anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update the sketch with a batch of centered embeddings."""

        if not torch.is_tensor(embeddings):
            raise TypeError("embeddings must be a torch.Tensor")
        if not self.enabled:
            return torch.empty(0, device=embeddings.device)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [B, D]")
        if embeddings.numel() == 0:
            return torch.empty(0, device=embeddings.device)

        with torch.no_grad():
            emb = embeddings.detach().to(torch.float32)
            batch, dim = emb.shape
            if mean is not None:
                self.set_mean(mean)
            if self._dim is None:
                self._dim = dim
            if self._dim != dim:
                raise ValueError("embedding dimensionality changed")
            if self._mean is None:
                raise RuntimeError(
                    "ThirdMomentSketch requires a mean vector; call set_mean() before update"
                )

            mu = self._mean.to(device=emb.device)
            proj = self._projection_matrix(device=emb.device, dim=dim)
            centered = emb - mu
            projected = centered @ proj
            projected_cubed = projected.pow(3)
            sum_proj = projected_cubed.sum(dim=0)
            count = torch.tensor(float(batch), device=proj.device, dtype=proj.dtype)
            if (
                dist_utils.get_world_size() > 1
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(sum_proj, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            avg_proj = sum_proj / count.clamp_min(1.0)
            self._update_ema(avg_proj)

            dirs = anchors if anchors is not None else centered
            if dirs.ndim != 2 or dirs.shape[1] != dim:
                raise ValueError("anchors must have shape [B, D]")
            dirs_f32 = dirs.detach().to(device=proj.device, dtype=proj.dtype)
            kappa = self.estimate(dirs_f32)
            clipped = self._clip_with_mad(kappa)
            self._record_summary(clipped)
            return clipped.detach()

    def estimate(self, anchors: torch.Tensor) -> torch.Tensor:
        """Estimate ``κ₃(x)`` for a set of anchor directions."""

        if self._projection is None or self._ema_u is None:
            raise RuntimeError("Sketch has not been initialized")
        if not torch.is_tensor(anchors):
            raise TypeError("anchors must be a torch.Tensor")
        if anchors.ndim != 2:
            raise ValueError("anchors must have shape [B, D]")
        if self._dim is not None and anchors.shape[1] != self._dim:
            raise ValueError("anchor dimensionality mismatch")

        proj = self._projection
        if proj.device != anchors.device:
            proj = proj.to(anchors.device)
            self._projection = proj
        weights = self._ema_u
        if weights.device != anchors.device:
            weights = weights.to(anchors.device)
            self._ema_u = weights
        coeffs = anchors.to(torch.float32) @ proj
        values = (coeffs.pow(3) * weights).sum(dim=1)
        return values

    def log_step(self, *, step: int, epoch: int) -> None:
        """Append a summary row to ``kappa3.csv`` if logging is enabled."""

        if (
            not self.enabled
            or not self.is_main
            or self._csv_writer is None
            or self._csv_file is None
            or self._last_summary is None
        ):
            return
        row = {
            "step": int(step),
            "epoch": int(epoch),
            "p50_abs": float(self._last_summary["p50_abs"]),
            "p90_abs": float(self._last_summary["p90_abs"]),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Close any open CSV handles."""

        if self._csv_file is not None:
            try:
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._csv_writer = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _projection_matrix(self, *, device: torch.device, dim: int) -> torch.Tensor:
        if self._projection is None:
            generator = self._get_generator(device)
            R = torch.randn(dim, self.rank, generator=generator, device=device)
            R = F.normalize(R, dim=0)
            self._projection = R.to(torch.float32)
        elif self._projection.device != device:
            self._projection = self._projection.to(device)
        return self._projection

    def _get_generator(self, device: torch.device) -> torch.Generator:
        torch_device = torch.device(device)
        key = str(torch_device)
        generator = self._rngs.get(key)
        if generator is None:
            generator = torch.Generator(device=torch_device)
            generator.manual_seed(self.seed)
            self._rngs[key] = generator
        return generator

    def _update_ema(self, value: torch.Tensor) -> None:
        if self._ema_u is None:
            self._ema_u = value.clone()
            return
        if self._ema_u.device != value.device:
            self._ema_u = self._ema_u.to(value.device)
        self._ema_u.mul_(self.ema_decay).add_(value, alpha=1.0 - self.ema_decay)

    def _clip_with_mad(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        median = torch.median(values)
        abs_dev = torch.abs(values - median)
        mad = torch.median(abs_dev)
        if float(mad) == 0.0:
            return values
        lower = (median - 5.0 * mad).item()
        upper = (median + 5.0 * mad).item()
        return values.clamp(min=lower, max=upper)

    def _record_summary(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            self._last_summary = None
            return
        abs_vals = torch.abs(values)
        p50 = torch.quantile(abs_vals, 0.5).item()
        p90 = torch.quantile(abs_vals, 0.9).item()
        self._last_summary = {"p50_abs": float(p50), "p90_abs": float(p90)}


_ACTIVE_SKETCH: Optional[ThirdMomentSketch] = None


def get_active_sketch() -> Optional[ThirdMomentSketch]:
    """Return the sketch registered for the current training step, if any."""

    return _ACTIVE_SKETCH


def _set_active_sketch(sketch: Optional[ThirdMomentSketch]) -> None:
    """Internal helper to update the active sketch reference."""

    global _ACTIVE_SKETCH
    _ACTIVE_SKETCH = sketch


__all__ = ["ThirdMomentSketch", "get_active_sketch", "_set_active_sketch"]
