"""Pipeline: shadow removal via learned U-Net or detector + CV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_removal import remove_shadows


def run_on_image(
    image_path: str | Path,
    output_path: str | Path,
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
    """Run shadow removal and write ``output_path``."""
    return remove_shadows(
        image_path,
        output_path,
        removal_backend=removal_backend,
        detection_weights=detection_weights,
        removal_weights=removal_weights,
        device=device,
        save_mask_path=save_mask_path,
        cv_method=cv_method,
        cv_kwargs=cv_kwargs,
        verbose=verbose,
    )
