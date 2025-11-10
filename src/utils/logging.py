"""Logging utilities for consistent experiment tracking."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_path: Optional[Path] = None) -> None:
    """Configure application-wide logging."""

    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
