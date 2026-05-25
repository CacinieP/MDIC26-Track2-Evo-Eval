"""
Configuration loading and validation utilities.

Handles:
- YAML config file loading
- Environment variable expansion (${VAR_NAME} syntax)
- Default configuration generation
- Config validation
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# Regex to find ${VAR} or ${VAR:-default} patterns
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in string values."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default)
            return os.environ.get(expr.strip(), match.group(0))
        return _ENV_VAR_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(path: str | Path | None = None) -> dict:
    """
    Load configuration from a YAML file with env var expansion.

    Args:
        path: Path to config YAML file. If None, tries:
              configs/config.yaml -> configs/config.example.yaml -> defaults

    Returns:
        Configuration dict with env vars expanded.
    """
    if path is None:
        candidates = [
            Path("configs/config.yaml"),
            Path("configs/config.local.yaml"),
            Path("configs/config.example.yaml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path is None:
        logger.info("No config file found, using defaults")
        return get_default_config()

    path = Path(path)
    if not path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return get_default_config()

    if not _HAS_YAML:
        logger.warning("PyYAML not installed, using defaults")
        return get_default_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = _expand_env_vars(config)
    logger.info(f"Config loaded from {path}")
    return config


def get_default_config() -> dict:
    """Return sensible default configuration."""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "timeout": 300,
        },
        "llm": {
            "planner": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            "verifier": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        },
        "mineru": {
            "device": "cuda",
            "table_enable": True,
            "formula_enable": True,
            "ocr_lang": "auto",
        },
        "pipeline": {
            "max_retries": 3,
            "max_concurrent_tasks": 4,
        },
        "storage": {
            "output_dir": "./data/output",
            "temp_dir": "./data/temp",
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/agent.log",
        },
    }


def validate_config(config: dict) -> list[str]:
    """
    Validate configuration and return list of issues found.

    Returns:
        List of issue description strings. Empty list means valid config.
    """
    issues: list[str] = []

    # Check server config
    server = config.get("server", {})
    port = server.get("port", 8000)
    if not isinstance(port, int) or not (1 <= port <= 65535):
        issues.append(f"Invalid server.port: {port}")

    # Check storage paths
    storage = config.get("storage", {})
    for key in ("output_dir", "temp_dir"):
        val = storage.get(key)
        if val and not isinstance(val, str):
            issues.append(f"storage.{key} should be a string path")

    # Check LLM config
    llm = config.get("llm", {})
    for role in ("planner", "verifier"):
        role_cfg = llm.get(role, {})
        provider = role_cfg.get("provider")
        if provider and provider not in ("anthropic", "openai", "openai_compatible", "local"):
            issues.append(f"Unknown LLM provider for {role}: {provider}")

    return issues
