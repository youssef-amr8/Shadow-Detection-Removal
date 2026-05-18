"""Full pipeline: shadow detection (ML) then shadow reduction (CV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_removal import remove_shadows


def run_on_image(
    image_path: str | Path,
    output_path: str | Path,
    *,
    detection_weights: str | Path | None = None,
    device: str | None = None,
    save_mask_path: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Detect shadows, apply CV removal, write ``output_path`` (and optional mask image)."""
    return remove_shadows(
        image_path,
        output_path,
        detection_weights=detection_weights,
        device=device,
        save_mask_path=save_mask_path,
        verbose=verbose,
    )
