"""Mixture moment estimators used for diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mfcl.utils import dist as dist_utils


class MixtureStats:
    """Maintain running estimates of Gaussian mixture moments."""

    _CSV_COLUMNS = ["step", "epoch", "K", "pi_min", "pi_max", "trace_B", "opnorm_B"]

    def __init__(
        self,
        *,
        K: int,
        assigner: str = "kmeans_online",
        mode: str = "ema",
        ema_decay: float = 0.95,
        enabled: bool = False,
        is_main: bool = True,
        cross_rank: bool = False,
        log_dir: Optional[str | Path] = None,
        store_scores: bool = False,
        scores_mode: str = "append",
        max_assign_iters: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.K = int(K)
        if self.K <= 0:
            raise ValueError("K must be positive")
        self.assigner = str(assigner)
        if self.assigner not in {"kmeans_online", "label_supervised"}:
            raise ValueError(f"Unsupported assigner: {assigner}")
        self.mode = str(mode)
        if self.mode not in {"per_batch", "ema", "label_supervised"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if not (0.0 < float(ema_decay) < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        self.ema_decay = float(ema_decay)
        self.enabled = bool(enabled)
        self.is_main = bool(is_main)
        self.cross_rank = bool(cross_rank)
        self.max_assign_iters = max(1, int(max_assign_iters))
        self.device = device
        self.dtype = dtype

        scores_mode_lc = scores_mode.lower()
        if scores_mode_lc not in {"append", "per_step"}:
            raise ValueError("scores_mode must be 'append' or 'per_step'")
        self._scores_mode = scores_mode_lc

        self._ema_pi: Optional[torch.Tensor] = None
        self._ema_mu: Optional[torch.Tensor] = None
        self._ema_sigma: Optional[torch.Tensor] = None
        self._ema_B: Optional[torch.Tensor] = None
        self._centroids: Optional[torch.Tensor] = None
        self._last_stats: Optional[Dict[str, torch.Tensor]] = None

        self._store_scores = bool(store_scores) and self.is_main
        self._scores_path: Optional[Path] = None
        self._scores_dir: Optional[Path] = None
        if self._store_scores:
            base = Path(log_dir) if log_dir is not None else None
            if base is None:
                raise ValueError("store_scores=True requires a log_dir")
            if self._scores_mode == "per_step":
                self._scores_dir = base.joinpath("mixture_scores")
                self._scores_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._scores_path = base.joinpath("mixture_scores.pt")

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        if log_dir is not None and self.enabled and self.is_main:
            self._csv_path = Path(log_dir).joinpath("mixture.csv")
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self._csv_path.open("a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_COLUMNS)
            if self._csv_path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._csv_file.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Update mixture statistics from the provided embeddings."""

        if not self.enabled:
            return {}
        if embeddings.numel() == 0:
            return {}

        with torch.no_grad():
            x = embeddings.detach()
            if self.device is not None:
                x = x.to(self.device)
            if self.dtype is not None:
                x = x.to(self.dtype)

            # Always operate in float32 for stability (bf16/half eigensolvers are fragile).
            x = x.to(torch.float32)

            labels_detached: Optional[torch.Tensor] = None
            if labels is not None:
                labels_detached = labels.detach()
                if labels_detached.device != x.device:
                    labels_detached = labels_detached.to(x.device)

            if self.cross_rank:
                x, labels_detached = self._maybe_gather_across_ranks(x, labels_detached)

            responsibilities = self._compute_responsibilities(x, labels_detached)
            stats = self._compute_moments(x, responsibilities)
            if self.mode == "ema":
                stats = self._apply_ema(stats)
            else:
                self._last_stats = {k: v.clone() for k, v in stats.items()}

            if self.assigner == "kmeans_online":
                self._centroids = stats["mu"].detach().clone()

            if self.assigner == "kmeans_online":
                stats = dict(stats)
                stats["R"] = responsibilities
                if self._last_stats is not None:
                    self._last_stats["R"] = responsibilities.clone()

            return stats

    def log_step(
        self,
        *,
        step: int,
        epoch: int,
        stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Append a summary row to ``mixture.csv``."""

        if (
            not self.enabled
            or not self.is_main
            or self._csv_writer is None
            or self._csv_file is None
        ):
            return
        if stats is None:
            stats = self._last_stats
        if stats is None:
            return

        pi = stats["pi"].to(torch.float32)
        B = stats["B"].to(torch.float32)
        trace_B = torch.trace(B).item()
        # B is symmetric PSD; use eigvalsh for a consistent spectral norm estimate.
        opnorm_B = torch.linalg.eigvalsh(B).abs().max().item()

        row = {
            "step": int(step),
            "epoch": int(epoch),
            "K": int(self.K),
            "pi_min": float(pi.min().item()),
            "pi_max": float(pi.max().item()),
            "trace_B": float(trace_B),
            "opnorm_B": float(opnorm_B),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._store_scores and "R" in stats:
            self._record_scores(step=step, epoch=epoch, responsibilities=stats["R"])

    def close(self) -> None:
        """Flush and close any open file handles."""

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
    def _compute_responsibilities(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.mode == "label_supervised" or self.assigner == "label_supervised":
            if labels is None:
                raise ValueError("Labels are required for label_supervised mode.")
            labels = labels.to(dtype=torch.long, device=embeddings.device)
            if labels.dim() != 1 or labels.shape[0] != embeddings.shape[0]:
                raise ValueError("Labels must be a vector aligned with embeddings")
            if labels.min() < 0 or labels.max() >= self.K:
                raise ValueError("Label ids must be in [0, K)")
            resp = F.one_hot(labels, num_classes=self.K).to(embeddings.dtype)
            return resp

        if self.assigner == "kmeans_online":
            return self._run_kmeans(embeddings)

        raise RuntimeError(f"Unsupported assigner {self.assigner}")

    def _run_kmeans(self, embeddings: torch.Tensor) -> torch.Tensor:
        n, d = embeddings.shape
        if self._centroids is None or self._centroids.shape != (self.K, d):
            perm = torch.randperm(n, device=embeddings.device)[: self.K]
            base = embeddings[perm].clone()
            if base.shape[0] < self.K:
                pad = embeddings[perm[:1]].repeat(self.K - base.shape[0], 1)
                base = torch.cat([base, pad], dim=0)
            self._centroids = base

        centroids = self._centroids.clone()
        for _ in range(self.max_assign_iters):
            distances = torch.cdist(embeddings, centroids, p=2)
            assignments = torch.argmin(distances, dim=1)
            for k in range(self.K):
                mask = assignments == k
                if mask.any():
                    centroids[k] = embeddings[mask].mean(dim=0)
        self._centroids = centroids
        resp = F.one_hot(assignments, num_classes=self.K).to(embeddings.dtype)
        return resp

    def _compute_moments(
        self, embeddings: torch.Tensor, responsibilities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        n, d = embeddings.shape
        weights = responsibilities.sum(dim=0)  # (K,)
        total = float(max(n, 1))
        pi = weights / total

        denom = weights.clamp(min=1e-6)
        weighted_sum = responsibilities.t().matmul(embeddings)
        mu = weighted_sum / denom.unsqueeze(1)

        empty = weights <= 1e-6
        if empty.any():
            mu = mu.clone()
            if self._centroids is not None and self._centroids.shape == mu.shape:
                mu[empty] = self._centroids[empty]
            else:
                mu[empty] = 0.0

        centered = embeddings.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.einsum("nk,nkd,nke->kde", responsibilities, centered, centered)
        cov = cov / denom.view(-1, 1, 1)

        if empty.any():
            cov = cov.clone()
            zero_cov = torch.zeros(
                (embeddings.size(1), embeddings.size(1)),
                device=cov.device,
                dtype=cov.dtype,
            )
            cov[empty] = zero_cov

        global_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
        mu_diff = mu - global_mu.unsqueeze(0)
        B = torch.einsum("k,kd,ke->de", pi, mu_diff, mu_diff)

        xBx = torch.einsum("nd,de,ne->n", embeddings, B, embeddings)
        if xBx.numel() > 0:
            median_xBx = torch.median(xBx)
        else:
            median_xBx = embeddings.new_tensor(0.0, dtype=B.dtype)

        global_cov = torch.einsum("k,kde->de", pi, cov)
        delta_sigma = cov - global_cov.unsqueeze(0)
        if delta_sigma.numel() > 0:
            if delta_sigma.dtype in {torch.float16, torch.bfloat16}:
                eigvals = torch.linalg.eigvalsh(delta_sigma.to(dtype=torch.float32))
            else:
                eigvals = torch.linalg.eigvalsh(delta_sigma)
            delta_sigma_max = eigvals.abs().max().to(dtype=B.dtype)
        else:
            delta_sigma_max = embeddings.new_tensor(0.0, dtype=B.dtype)

        stats = {
            "pi": pi,
            "mu": mu,
            "Sigma": cov,
            "B": B,
            "median_xBx": median_xBx,
            "delta_sigma_max": delta_sigma_max,
        }
        self._last_stats = {k: v.clone() for k, v in stats.items()}
        return stats

    def _apply_ema(self, stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decay = self.ema_decay
        extras = {k: v.clone() for k, v in stats.items() if k not in {"pi", "mu", "Sigma", "B"}}
        if self._ema_pi is None:
            self._ema_pi = stats["pi"].clone()
            self._ema_mu = stats["mu"].clone()
            self._ema_sigma = stats["Sigma"].clone()
            self._ema_B = stats["B"].clone()
        else:
            self._ema_pi = decay * self._ema_pi + (1.0 - decay) * stats["pi"]
            self._ema_mu = decay * self._ema_mu + (1.0 - decay) * stats["mu"]
            self._ema_sigma = decay * self._ema_sigma + (1.0 - decay) * stats["Sigma"]
            self._ema_B = decay * self._ema_B + (1.0 - decay) * stats["B"]
        ema_stats = {
            "pi": self._ema_pi.clone(),
            "mu": self._ema_mu.clone(),
            "Sigma": self._ema_sigma.clone(),
            "B": self._ema_B.clone(),
        }
        if extras:
            ema_stats.update(extras)
        self._last_stats = {k: v.clone() for k, v in ema_stats.items()}
        return ema_stats

    def _record_scores(
        self,
        *,
        step: int,
        epoch: int,
        responsibilities: torch.Tensor,
    ) -> None:
        if self._scores_dir is not None:
            path = self._scores_dir.joinpath(f"step_{int(step):06d}.pt")
            torch.save(
                {
                    "step": int(step),
                    "epoch": int(epoch),
                    "responsibilities": responsibilities.cpu(),
                },
                path,
            )
            return

        if self._scores_path is None:
            return
        entry = {
            "step": int(step),
            "epoch": int(epoch),
            "responsibilities": responsibilities.cpu(),
        }
        existing: Iterable[dict] = []
        if self._scores_path.exists():
            try:
                existing_obj = torch.load(self._scores_path)
                if isinstance(existing_obj, list):
                    existing = existing_obj
            except Exception:
                existing = []
        data = list(existing)
        data.append(entry)
        torch.save(data, self._scores_path)

    def _maybe_gather_across_ranks(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not dist_utils.is_dist() or dist_utils.get_world_size() <= 1:
            return embeddings, labels

        gathered_embeddings = self._all_gather_variable(embeddings)
        gathered_labels: Optional[torch.Tensor] = None
        if labels is not None:
            gathered_labels = self._all_gather_variable(labels.to(torch.long))
        return gathered_embeddings, gathered_labels

    def _all_gather_variable(self, tensor: torch.Tensor) -> torch.Tensor:
        world = dist_utils.get_world_size()
        if world <= 1:
            return tensor

        local_count = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        counts = [torch.zeros_like(local_count) for _ in range(world)]
        dist.all_gather(counts, local_count)
        sizes = [int(c.item()) for c in counts]
        max_count = max(sizes, default=0)
        if max_count == 0:
            return tensor.new_zeros((0,) + tensor.shape[1:])

        if tensor.shape[0] < max_count:
            pad_shape = (max_count - tensor.shape[0],) + tensor.shape[1:]
            padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
            padded = torch.cat([tensor, padding], dim=0)
        else:
            padded = tensor

        gather_list = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(gather_list, padded)

        pieces = [gather_list[idx][: sizes[idx]] for idx in range(world) if sizes[idx] > 0]
        if not pieces:
            return tensor.new_zeros((0,) + tensor.shape[1:])
        return torch.cat(pieces, dim=0)
