"""Shadow removal: detector → mask → U-Net, or detector → mask → CV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from shadow_detector import ShadowDetector, resolve_detection_weights_path
from shadow_removal_cv import apply_shadow_removal
from shadow_removal_net import ShadowRemovalNet, resolve_removal_weights_path


def remove_shadows_unet(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    detection_weights: str | Path | None = None,
    removal_weights: str | Path | None = None,
    device: str | None = None,
    save_mask_path: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Full ML pipeline: detection U-Net → mask → removal U-Net on original image.

    Removal was trained with (train_A RGB + train_B mask) → train_C.
    At inference, train_B is replaced by the detector mask.
    """
    path = Path(image_path)
    if verbose:
        dw = resolve_detection_weights_path(detection_weights)
        rw = resolve_removal_weights_path(removal_weights)
        print(f"[1/5] Detector weights: {dw.resolve()}")
        print(f"[2/5] Removal U-Net weights: {rw.resolve()}")
        print(f"[3/5] Loading image: {path.resolve()}")

    detector = ShadowDetector(detection_weights, device=device)
    remover = ShadowRemovalNet(removal_weights, device=device)

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if verbose:
        print("[4/5] Detection U-Net → shadow mask …")
    mask = detector.predict_mask(rgb)

    if verbose:
        print("[5/5] Removal U-Net (image + mask) → shadow-free …")
    out_rgb = remover.predict(rgb, mask)

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
            print(f"      Saved: {outp.resolve()}")

    return out_rgb


def remove_shadows(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    removal_backend: str = "unet",
    detection_weights: str | Path | None = None,
    removal_weights: str | Path | None = None,
    device: str | None = None,
    save_mask_path: str | Path | None = None,
    cv_method: str = "lab-pro",
    cv_kwargs: dict | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Shadow removal on one image.

    removal_backend:
      - ``unet`` (default): detector mask → removal U-Net
      - ``cv``: detector mask → classical OpenCV
    """
    backend = removal_backend.lower().strip()
    if backend in ("unet", "ml", "learned", "model"):
        return remove_shadows_unet(
            image_path,
            output_path,
            detection_weights=detection_weights,
            removal_weights=removal_weights,
            device=device,
            save_mask_path=save_mask_path,
            verbose=verbose,
        )

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
        print(f"[3/4] U-Net mask → CV removal (method={cv_method}) …")
    mask = detector.predict_mask(rgb)
    cv_kw = dict(cv_kwargs or {})
    if verbose and cv_method in ("lab-pro", "illumination", "pro", "professional"):
        cv_kw["verbose"] = True
    out_rgb = apply_shadow_removal(rgb, mask, method=cv_method, **cv_kw)

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
