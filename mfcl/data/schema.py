"""Utilities for validating and normalizing batch schema."""

from __future__ import annotations

from typing import Any, Mapping

import torch


CANONICAL_TARGET_KEY = "target"
_LEGACY_LABEL_KEYS = ("mixture_labels", "labels", "label")


def extract_labels(
    batch: Mapping[str, Any], *, required: bool = False
) -> torch.Tensor | None:
    """Return the canonical label tensor from a collated batch.

    Args:
        batch: Mapping representing a collated batch.
        required: When ``True``, raise if the canonical key is missing.

    Returns:
        The label tensor stored under the canonical ``"target"`` key or ``None``
        when the key is absent and ``required`` is ``False``.

    Raises:
        TypeError: If ``batch`` is not a mapping or the value is not a tensor.
        KeyError: If legacy label keys are encountered or the canonical key is
            missing when ``required`` is ``True``.
        ValueError: If the canonical key is present but the value is ``None``
            when ``required`` is ``True``.
    """

    if not isinstance(batch, Mapping):
        raise TypeError(
            "extract_labels expects the batch to be a mapping of tensors"
        )

    if CANONICAL_TARGET_KEY in batch:
        value = batch[CANONICAL_TARGET_KEY]
        if value is None:
            if required:
                raise ValueError(
                    "Batch['target'] is None but supervised labels are required"
                )
            return None
        if not torch.is_tensor(value):
            raise TypeError(
                "Batch['target'] must be a torch.Tensor when provided"
            )
        return value

    legacy_keys = [key for key in _LEGACY_LABEL_KEYS if key in batch]
    if legacy_keys:
        aliases = ", ".join(f"'{key}'" for key in legacy_keys)
        raise KeyError(
            f"Batch provides label key(s) {aliases} but the canonical key is 'target'. "
            "Update your dataset or collate function to emit the canonical key."
        )

    if required:
        available = ", ".join(sorted(batch.keys())) or "<empty>"
        raise KeyError(
            f"Batch is missing required '{CANONICAL_TARGET_KEY}' labels. "
            f"Available keys: {available}."
        )

    return None


__all__ = ["CANONICAL_TARGET_KEY", "extract_labels"]

