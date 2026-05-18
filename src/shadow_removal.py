"""Detect shadows (U-Net) then remove them with classical CV (LAB luminance)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from shadow_detector import ShadowDetector, resolve_detection_weights_path
from shadow_removal_cv import remove_shadows_lab


def remove_shadows(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    detection_weights: str | Path | None = None,
    device: str | torch.device | None = None,
    save_mask_path: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load image → detector mask → LAB-based shadow reduction → optional saves.

    Returns:
        RGB float32 (H, W, 3) in [0, 1].
    """
    path = Path(image_path)
    if verbose:
        wpath = resolve_detection_weights_path(detection_weights)
        print(f"[1/4] Shadow detector weights: {wpath.resolve()}")
        print(f"[2/4] Loading image: {path.resolve()}")

    detector = ShadowDetector(detection_weights, device=device)

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if verbose:
        print("[3/4] Running U-Net → shadow mask, then LAB (CV) shadow reduction …")
    mask = detector.predict_mask(rgb)
    out_rgb = remove_shadows_lab(rgb, mask)

    if save_mask_path is not None:
        mp = Path(save_mask_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        m = mask if mask.ndim == 2 else mask[:, :, 0]
        cv2.imwrite(str(mp), (np.clip(m, 0, 1) * 255.0).astype(np.uint8))
        if verbose:
            print(f"      Saved mask: {mp.resolve()}")

    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_bgr = cv2.cvtColor(
            (out_rgb * 255.0).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(outp), out_bgr)
        if verbose:
            print(f"[4/4] Saved shadow-reduced image: {outp.resolve()}")

    return out_rgb
