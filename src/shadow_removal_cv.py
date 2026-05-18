"""Classical (non-learned) shadow reduction using the mask + LAB luminance matching."""

from __future__ import annotations

import cv2
import numpy as np


def remove_shadows_lab(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """
    Reduce shadows by matching shadow-region lightness statistics to well-lit pixels.

    Args:
        rgb: float32 (H, W, 3) in [0, 1], RGB.
        mask: float32 (H, W) or (H, W, 1) in [0, 1]; higher = shadow (same convention as detector).
        blur_sigma: Gaussian sigma applied to the mask for softer transitions.

    Returns:
        float32 RGB in [0, 1], same shape as ``rgb``.
    """
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = np.clip(m, 0.0, 1.0)

    k = int(max(3, min(31, 2 * int(round(3 * blur_sigma)) + 1)))
    if k % 2 == 0:
        k += 1
    m_soft = cv2.GaussianBlur(m, (k, k), sigmaX=blur_sigma, sigmaY=blur_sigma)

    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, Bc = cv2.split(lab)
    lf = L.astype(np.float32)

    # Weighted "lit" / "shadow" statistics from soft mask
    w_lit = np.clip(1.0 - m_soft, 0.0, 1.0)
    w_sh = np.clip(m_soft, 0.0, 1.0)
    sum_lit = float(np.sum(w_lit) + 1e-6)
    sum_sh = float(np.sum(w_sh) + 1e-6)

    # If the whole frame is flagged as shadow, fall back to brightest pixels as "lit"
    if sum_lit < 500.0:
        thr = float(np.percentile(m_soft, 70))
        w_lit = np.clip(thr - m_soft, 0.0, 1.0)
        sum_lit = float(np.sum(w_lit) + 1e-6)

    mu_lit = np.sum(lf * w_lit) / sum_lit
    var_lit = np.sum(w_lit * (lf - mu_lit) ** 2) / sum_lit
    sig_lit = float(np.sqrt(max(var_lit, 1e-6)))

    mu_sh = np.sum(lf * w_sh) / sum_sh
    var_sh = np.sum(w_sh * (lf - mu_sh) ** 2) / sum_sh
    sig_sh = float(np.sqrt(max(var_sh, 1e-6)))

    # Affine luminance transfer toward lit-region statistics
    scale = sig_lit / (sig_sh + 1e-3)
    lf_corr = (lf - mu_sh) * scale + mu_lit
    lf_out = (1.0 - m_soft) * lf + m_soft * lf_corr
    lf_out = np.clip(lf_out, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([lf_out, A, Bc])
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return (rgb_out.astype(np.float32) / 255.0).clip(0.0, 1.0)
