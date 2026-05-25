"""
Logger configuration - Structured logging with loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(config: dict | None = None):
    """Configure structured logging."""
    config = config or {}
    level = config.get("level", "INFO")
    log_format = config.get("format", "json")
    log_file = config.get("file", "./logs/agent.log")

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        colorize=True,
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation=config.get("rotation", "100 MB"),
        retention=config.get("retention", "30 days"),
        serialize=(log_format == "json"),
    )

    logger.info("Logging initialized")
    return logger
