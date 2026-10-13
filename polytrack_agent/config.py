"""Load and merge the vision-agent config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

OUTPUT_DENYLIST = frozenset(
    {
        "r",
        "enter",
        "return",
        "esc",
        "escape",
        "f6",
        "f7",
        "f8",
        "tab",
        "alt",
        "ctrl",
        "cmd",
        "super",
        "win",
        "menu",
    }
)

_PACKAGE_CONFIG = Path(__file__).resolve().parent / "default_config.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_output_keys(keys: list[str]) -> list[str]:
    """Normalize driving keys and reject restart, pause, and modifier keys."""
    normalized = [str(key).lower().strip() for key in keys]
    if not normalized or any(not key for key in normalized):
        raise ValueError("record.keys must list at least one driving key")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate keys in record.keys: {normalized}")
    denied = [key for key in normalized if key in OUTPUT_DENYLIST]
    if denied:
        raise ValueError(
            f"Refusing output keys {denied}. The agent must not press restart, "
            "pause, or modifier keys. Restart the track yourself, or let a later "
            "RL wrapper call an explicit restart hook."
        )
    return normalized


def default_config_path() -> Path:
    if _PACKAGE_CONFIG.is_file():
        return _PACKAGE_CONFIG
    fallback = _PACKAGE_CONFIG.parents[1] / "configs" / "default.yaml"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("default_config.yaml is missing from the package")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with default_config_path().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {default_config_path()} is empty")
    if path is not None:
        overlay_path = Path(path)
        with overlay_path.open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        if not isinstance(overlay, dict):
            raise ValueError(f"Config {overlay_path} must be a mapping")
        config = deep_merge(config, overlay)
    config["record"]["keys"] = validate_output_keys(list(config["record"]["keys"]))
    return config
