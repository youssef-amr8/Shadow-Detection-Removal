"""Load trained shadow-detection U-Net and predict shadow masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from model import UNet
from utils import load_torch_checkpoint, repo_root


def resolve_detection_weights_path(weights_path: str | Path | None) -> Path:
    """
    Single-file PyTorch checkpoints only (``.pth`` / ``.pt``).

    Tries, in order: explicit path, ``models/shadow_detection.pth``, repo-root
    ``shadow_detection.pth``. A directory named ``shadow_detection/`` with only
    ``version`` / ``byteorder`` is **not** a loadable checkpoint (often a partial
    unzip of a ``.pth`` zip); use the original one-file save from training.
    """
    root = repo_root()
    if weights_path is not None:
        p = Path(weights_path)
        if p.is_dir():
            raise FileNotFoundError(
                f"Detection weights must be a file, not a directory: {p.resolve()}\n"
                "Use the .pth from training (e.g. models/shadow_detection.pth)."
            )
        if not p.is_file():
            raise FileNotFoundError(
                f"Detection weights not found: {p.resolve()}\n"
                "Expected a single .pth/.pt file (state_dict), e.g. models/shadow_detection.pth"
            )
        return p

    defaults = (
        root / "models" / "shadow_detection.pth",
        root / "shadow_detection.pth",
    )
    for candidate in defaults:
        if candidate.is_file():
            return candidate

    tried_lines = "\n".join(f"  - {c.resolve()}" for c in defaults)

    hint = ""
    meta = root / "shadow_detection" / "version"
    if meta.is_file():
        hint = (
            "\nNote: A `shadow_detection/` folder with only `version`/`byteorder` is not "
            "the full model file. Training saves `models/shadow_detection.pth` — use that "
            "file (or pass --detector-weights)."
        )

    raise FileNotFoundError(
        "No detection checkpoint found. Default search paths were:\n"
        f"{tried_lines}\n\n"
        "Fix one of:\n"
        "  • Train and save: python main.py train-detection --data data/ISTD_Dataset\n"
        "  • Or pass your checkpoint file:\n"
        "      python main.py run --image YOUR.jpg --output out.png "
        "--detector-weights /absolute/path/to/shadow_detection.pth\n"
        + hint
    )


class ShadowDetector:
    """Runs the detection U-Net (1-channel sigmoid mask)."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        img_size: int = 256,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_size = img_size
        self.model = UNet().to(self.device)
        path = resolve_detection_weights_path(weights_path)
        state = load_torch_checkpoint(path, self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            image_rgb: float32 or uint8, shape (H, W, 3), RGB.

        Returns:
            float32 mask (H, W, 1) in [0, 1], same spatial size as input.
        """
        if image_rgb.dtype != np.float32:
            image_rgb = image_rgb.astype(np.float32)
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

        h, w = image_rgb.shape[:2]
        small = cv2.resize(
            image_rgb,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        x = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).to(self.device)
        m = self.model(x).cpu().numpy()[0, 0]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m.astype(np.float32)[..., np.newaxis]

    def predict_mask_from_path(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load BGR image from disk; returns (rgb_float_hw3, mask_hw1)."""
        path = Path(image_path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb, self.predict_mask(rgb)
