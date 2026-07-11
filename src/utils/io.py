from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)
    return path


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge updates into base and return base."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path, default_path: str | Path | None = "configs/default.yaml") -> dict[str, Any]:
    """Load default config then overlay a run-specific config."""
    config_path = Path(config_path)
    if default_path and Path(default_path).exists() and Path(default_path).resolve() != config_path.resolve():
        cfg = load_yaml(default_path)
        deep_update(cfg, load_yaml(config_path))
        return cfg
    return load_yaml(config_path)


def copy_or_symlink(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    """Prefer symlinks for large image datasets and fall back to copying."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return dst
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return dst


def copy_file(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path
