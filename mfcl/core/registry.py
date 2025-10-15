"""Lightweight string-to-callable registry with explicit add/get.

The Registry class maps normalized string keys to callables or classes.
No implicit imports, no global singletons, and helpful error messages.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List
import re


class Registry:
    """String-key registry with explicit add/get and helpful errors.

    Keys are normalized to a restricted character set to avoid ambiguity and
    import-order fragility. Allowed characters are lowercase letters, digits,
    and dashes (``[a-z0-9-]``). Keys must be provided already normalized.
    """

    _KEY_PATTERN = re.compile(r"^[a-z0-9-]+$")

    def __init__(self, name: str) -> None:
        """Initialize an empty registry.

        Args:
            name: Human-readable name used in error messages.
        """
        if not isinstance(name, str) or not name:
            raise TypeError("Registry name must be a non-empty string.")
        self.name = name
        self._map: Dict[str, Callable[..., Any]] = {}

    def _validate_key(self, key: str) -> None:
        """Validate key format and normalization.

        Args:
            key: Candidate key.

        Raises:
            TypeError: If key is not a string.
            ValueError: If key is empty or not normalized/allowed.
        """
        if not isinstance(key, str):
            raise TypeError("Registry key must be a string.")
        if not key:
            raise ValueError("Registry key must be non-empty.")
        if key != key.lower():
            raise ValueError(
                f"Registry key '{key}' must be lowercase; got non-normalized key."
            )
        if not self._KEY_PATTERN.match(key):
            raise ValueError(
                f"Registry key '{key}' contains invalid characters. Use only [a-z0-9-]."
            )

    def add(self, key: str, obj: Callable[..., Any], *, strict: bool = True) -> None:
        """Register ``obj`` under ``key``.

        Args:
            key: Unique identifier (lowercase, alnum + dash allowed).
            obj: Class or callable (constructor) to instantiate later.
            strict: If ``True`` (default), refuse to overwrite existing keys.
                Set to ``False`` to replace an existing registration explicitly.

        Raises:
            TypeError: If key is not str or obj is not callable/class.
            ValueError: If key is empty or not normalized.
            KeyError: If key already exists and ``strict`` is ``True``.
        """
        self._validate_key(key)
        if not callable(obj):
            raise TypeError(
                f"Object registered under key '{key}' must be callable or a class."
            )
        if key in self._map and strict:
            raise KeyError(
                f"Key '{key}' already exists in '{self.name}' registry; pass strict=False to overwrite."
            )
        self._map[key] = obj

    def has(self, key: str) -> bool:
        """Return True if ``key`` exists in registry."""
        return key in self._map

    def __contains__(self, key: str) -> bool:
        """Alias for :meth:`has` enabling ``key in registry`` syntax."""
        return self.has(key)

    def __len__(self) -> int:
        """Return the number of registered keys."""
        return len(self._map)

    def __repr__(self) -> str:
        """Debug-friendly representation showing registry name and size."""
        return f"Registry(name={self.name!r}, keys={len(self._map)})"

    def get(self, key: str) -> Callable[..., Any]:
        """Retrieve a previously registered object.

        Args:
            key: Identifier used in :meth:`add`.

        Returns:
            Registered callable/class.

        Raises:
            KeyError: If key not found. The message includes the registry name
                and up to 50 available keys for quick discovery.
        """
        try:
            return self._map[key]
        except KeyError as e:
            keys: List[str] = sorted(self._map.keys())
            hint = ", ".join(keys[:50]) if keys else "<empty>"
            raise KeyError(
                f"Key '{key}' not found in '{self.name}' registry. Available: {hint}"
            ) from e

    def keys(self) -> List[str]:
        """Return a sorted list of registered keys."""
        return sorted(self._map.keys())


__all__ = ["Registry"]
