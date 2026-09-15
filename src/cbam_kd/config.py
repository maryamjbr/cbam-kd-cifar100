"""YAML configuration helpers with explicit validation errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    data["_config_path"] = str(config_path.resolve())
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a mapping")
    return value
