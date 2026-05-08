"""Utility helpers for shadow-removal-detection."""

from pathlib import Path


def ensure_dir(path: Path):
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
