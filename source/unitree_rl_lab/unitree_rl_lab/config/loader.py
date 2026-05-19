from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(value: Any) -> Any:
    """Recursively convert dict/list to dot-access objects."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load YAML config as a nested dot-access object (strict mode)."""
    if path is not None:
        cfg_path = Path(path)
    else:
        # 固定读取项目根目录配置，避免“读错文件但不报错”。
        cfg_path = Path(__file__).resolve().parents[4] / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")

    return _to_namespace(raw)
