"""Utility helpers for shadow-removal-detection."""

from pathlib import Path


def repo_root() -> Path:
    """Project root (parent of ``src``)."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path):
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_torch_checkpoint(path: Path | str, map_location):
    """Load a ``state_dict`` file; compatible with PyTorch versions before ``weights_only``."""
    import torch

    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)
