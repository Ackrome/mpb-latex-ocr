"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(config_paths: list[str] | None, overrides: list[str] | None = None) -> dict[str, Any]:
    config_paths = config_paths or ["configs/train.yaml"]
    cfg = OmegaConf.create()
    for path in config_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Top-level config must be a mapping.")
    return resolved


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, name))
        elif value is None:
            flat[name] = "null"
        elif isinstance(value, (str, int, float, bool)):
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat
