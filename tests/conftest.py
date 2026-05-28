"""
Shared fixtures for MinerU DataAgent tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def sample_config():
    """Return a minimal valid config dict."""
    return {
        "server": {"host": "127.0.0.1", "port": 8000, "timeout": 60},
        "mineru": {"device": "cpu", "table_enable": True, "formula_enable": False},
        "pipeline": {"max_retries": 2, "max_concurrent_tasks": 2, "quality_threshold": 0.7},
        "storage": {"output_dir": "./data/output", "temp_dir": "./data/temp"},
        "logging": {"level": "WARNING"},
    }


@pytest.fixture
def tool_registry(sample_config):
    """Return a tool registry with all tools instantiated."""
    from src.tools import create_tool_registry
    return create_tool_registry(sample_config)
