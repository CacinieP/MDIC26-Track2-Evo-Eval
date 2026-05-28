"""
Tests for configuration loading and validation.
Tests: load_config, get_default_config, validate_config, _expand_env_vars
"""

import os
from pathlib import Path

import pytest

from src.utils.config import (
    _expand_env_vars,
    get_default_config,
    load_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# _expand_env_vars
# ---------------------------------------------------------------------------

class TestExpandEnvVars:
    """Tests for environment variable expansion in config values."""

    def test_no_env_vars(self):
        result = _expand_env_vars("hello world")
        assert result == "hello world"

    def test_simple_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_123", "my_secret")
        result = _expand_env_vars("${TEST_KEY_123}")
        assert result == "my_secret"

    def test_env_var_with_default(self, monkeypatch):
        # Ensure the var does NOT exist
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        result = _expand_env_vars("${NONEXISTENT_VAR_XYZ:-fallback_value}")
        assert result == "fallback_value"

    def test_env_var_default_ignored_when_set(self, monkeypatch):
        monkeypatch.setenv("EXISTING_VAR_ABC", "actual")
        result = _expand_env_vars("${EXISTING_VAR_ABC:-unused}")
        assert result == "actual"

    def test_nested_dict_expansion(self, monkeypatch):
        monkeypatch.setenv("MY_HOST", "localhost")
        monkeypatch.setenv("MY_PORT", "5432")
        config = {
            "server": {
                "host": "${MY_HOST}",
                "port": "${MY_PORT}",
            }
        }
        result = _expand_env_vars(config)
        assert result["server"]["host"] == "localhost"
        assert result["server"]["port"] == "5432"

    def test_list_expansion(self, monkeypatch):
        monkeypatch.setenv("ITEM_A", "alpha")
        result = _expand_env_vars(["${ITEM_A}", "static_value"])
        assert result == ["alpha", "static_value"]

    def test_non_string_passthrough(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_unresolved_env_var_kept(self):
        """Unresolved ${VAR} should remain as-is when var not in env."""
        result = _expand_env_vars("${COMPLETELY_UNKNOWN_VAR_ZZZ}")
        assert "${COMPLETELY_UNKNOWN_VAR_ZZZ}" in result


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    """Tests for default configuration generation."""

    def test_returns_dict(self):
        cfg = get_default_config()
        assert isinstance(cfg, dict)

    def test_has_required_sections(self):
        cfg = get_default_config()
        assert "server" in cfg
        assert "llm" in cfg
        assert "mineru" in cfg
        assert "pipeline" in cfg
        assert "storage" in cfg
        assert "logging" in cfg

    def test_server_defaults(self):
        cfg = get_default_config()
        assert cfg["server"]["port"] == 8000
        assert cfg["server"]["host"] == "0.0.0.0"

    def test_mineru_device_default(self):
        cfg = get_default_config()
        assert cfg["mineru"]["device"] in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for config validation."""

    def test_valid_config_no_issues(self):
        cfg = get_default_config()
        issues = validate_config(cfg)
        assert issues == []

    def test_invalid_port(self):
        cfg = get_default_config()
        cfg["server"]["port"] = 99999
        issues = validate_config(cfg)
        assert any("port" in issue.lower() for issue in issues)

    def test_invalid_provider(self):
        cfg = get_default_config()
        cfg["llm"]["planner"]["provider"] = "invalid_provider"
        issues = validate_config(cfg)
        assert any("provider" in issue.lower() for issue in issues)

    def test_empty_config_no_crash(self):
        issues = validate_config({})
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_example_config(self):
        """Load the shipped config.example.yaml."""
        example = Path(__file__).resolve().parent.parent / "configs" / "config.example.yaml"
        if not example.exists():
            pytest.skip("config.example.yaml not found")
        cfg = load_config(str(example))
        assert isinstance(cfg, dict)
        assert "server" in cfg

    def test_load_missing_file_returns_default(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, dict)
        assert "server" in cfg

    def test_load_none_returns_config(self, tmp_path, monkeypatch):
        """load_config(None) should find a config file or return defaults."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config(None)
        assert isinstance(cfg, dict)
