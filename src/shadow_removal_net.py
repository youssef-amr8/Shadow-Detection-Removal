"""Removal U-Net: (original RGB + shadow mask) → shadow-free RGB."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from model import UNet
from utils import load_torch_checkpoint, repo_root


def resolve_removal_weights_path(weights_path: str | Path | None) -> Path:
    root = repo_root()
    if weights_path is not None:
        p = Path(weights_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Removal weights not found: {p.resolve()}\n"
                "Expected models/shadow_removal.pth from training."
            )
        return p

    for candidate in (
        root / "models" / "shadow_removal.pth",
        root / "shadow_removal.pth",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No shadow removal checkpoint found. Train first:\n"
        "  python main.py train-removal --data data/ISTD_Dataset\n"
        "or pass --removal-weights to your .pth file."
    )


class ShadowRemovalNet:
    """U-Net: 4ch input (RGB + mask) → 3ch clean RGB."""

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
        self.model = UNet(in_channels=4, out_channels=3).to(self.device)
        path = resolve_removal_weights_path(weights_path)
        state = load_torch_checkpoint(path, self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.weights_path = path

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            image_rgb: (H, W, 3) RGB float [0,1] or uint8.
            mask: (H, W) or (H, W, 1) float [0,1], shadow=high.

        Returns:
            float32 RGB (H, W, 3) in [0, 1], same size as input.
        """
        if image_rgb.dtype != np.float32:
            image_rgb = image_rgb.astype(np.float32)
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = np.clip(m, 0.0, 1.0)

        h, w = image_rgb.shape[:2]
        small_rgb = cv2.resize(
            image_rgb,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        small_m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        x = np.concatenate([small_rgb, small_m[..., np.newaxis]], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(x).cpu().numpy()[0].transpose(1, 2, 0)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
