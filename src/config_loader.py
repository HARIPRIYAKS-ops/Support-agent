"""
src/config_loader.py
Centralised configuration loader.  Reads config.yaml (repo-relative defaults),
then overrides with env vars from .env / environment.

Usage:
    from src.config_loader import cfg, resolve_path
    raw_path = resolve_path(cfg["paths"]["raw_data"])
"""

import os
import sys
import yaml
from pathlib import Path

# ── Repo root (parent of this file's directory) ──────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    config_path = _REPO_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment-variable overrides to the config dict."""
    # Load .env if present (won't error if missing)
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass  # python-dotenv optional; fall back to real env vars only

    raw_path = os.environ.get("RAW_DATA_PATH", "").strip()
    if raw_path:
        config["paths"]["raw_data"] = raw_path

    return config


cfg: dict = _apply_env_overrides(_load_config())


def resolve_path(relative_or_absolute: str) -> Path:
    """
    Return an absolute Path.
    If already absolute, return as-is; otherwise resolve relative to repo root.
    """
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def get_raw_data_path() -> Path:
    """Convenience wrapper — always returns the resolved raw CSV path."""
    return resolve_path(cfg["paths"]["raw_data"])
