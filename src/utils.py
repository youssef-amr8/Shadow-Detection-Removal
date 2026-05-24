from pathlib import Path
import torch

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_torch_checkpoint(path: Path | str, map_location):
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)
