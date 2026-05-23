"""Classical shadow reduction: professional LAB pipeline + legacy methods."""

from __future__ import annotations

import cv2
import numpy as np

CV_METHODS = ("lab-pro", "illumination", "hsv", "lab")


def _prepare_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb, rgb_u8


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    return np.clip(m, 0.0, 1.0)


def _feather_mask(mask: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    m = _normalize_mask(mask)
    kw, kh = ksize
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    return cv2.GaussianBlur(m, (kw, kh), 0)


def _lit_shadow_means(
    channel: np.ndarray,
    mask_alpha: np.ndarray,
) -> tuple[float, float]:
    w_sh = mask_alpha
    w_lit = np.clip(1.0 - mask_alpha, 0.0, 1.0)
    sum_lit = float(np.sum(w_lit) + 1e-6)
    sum_sh = float(np.sum(w_sh) + 1e-6)
    if sum_lit < 500.0:
        thr = float(np.percentile(mask_alpha, 70))
        w_lit = np.clip(thr - mask_alpha, 0.0, 1.0)
        sum_lit = float(np.sum(w_lit) + 1e-6)
    mu_lit = float(np.sum(channel * w_lit) / sum_lit)
    mu_sh = float(np.sum(channel * w_sh) / sum_sh)
    return mu_lit, mu_sh


def _single_scale_retinex_L(L: np.ndarray, sigma: float = 80.0, strength: float = 0.25) -> np.ndarray:
    """Mild SSR on L to separate illumination; blended with original L."""
    Lf = L.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(Lf, (0, 0), sigma)
    ret = np.log(Lf) - np.log(blur + 1.0)
    rmin, rmax = float(ret.min()), float(ret.max())
    if rmax - rmin > 1e-6:
        ret_n = (ret - rmin) / (rmax - rmin) * 255.0
    else:
        ret_n = Lf - 1.0
    out = (1.0 - strength) * Lf + strength * ret_n
    return np.clip(out, 0, 255).astype(np.float32)


def _odd_ksize(ksize: tuple[int, int]) -> tuple[int, int]:
    kw, kh = ksize
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    return kw, kh


def remove_shadows_lab_pro(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mask_blur_blend: tuple[int, int] = (31, 31),
    mask_dilate_iter: int = 0,
    blend_power: float = 1.0,
    shadow_threshold: float = 0.45,
    brightness_mode: str = "match",
    brightness_boost: float = 1.0,
    max_ratio: float = 5.0,
    min_ratio: float = 1.5,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    use_clahe: bool = False,
    match_chroma: bool = True,
    chroma_strength: float = 0.65,
    use_retinex: bool = False,
    retinex_sigma: float = 80.0,
    retinex_strength: float = 0.25,
    verbose: bool = False,
) -> np.ndarray:
    """
    LAB illumination transfer with seamless feather (no white/dark borders).

    Uses per-pixel soft lift inside the (feathered) mask only — no mask dilation,
    cap at mean lit L so edges do not overshoot into a bright ring.

    Manual brightness knobs (strongest → weakest effect):
      brightness_boost   — scales lift / ratio (try 1.0–1.4)
      max_ratio          — cap for ratio mode (try 3.0–5.0)
      min_ratio          — floor for ratio mode (try 1.3–1.6)
      brightness_mode    — "match" (additive) or "ratio" (multiply L)
    Presets live in main.py → _lab_pro_kwargs().
    """
    rgb, rgb_u8 = _prepare_rgb(rgb)
    m = _normalize_mask(mask)
    m_u8 = (m * 255.0).astype(np.uint8)

    if mask_dilate_iter > 0:
        k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m_u8 = cv2.dilate(m_u8, k_morph, iterations=mask_dilate_iter)

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, Bc = cv2.split(lab)
    L_work = L.astype(np.float32)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        L_work = clahe.apply(L_work.astype(np.uint8)).astype(np.float32)

    if use_retinex:
        L_work = _single_scale_retinex_L(L_work, sigma=retinex_sigma, strength=retinex_strength)

    shadow_bool = m > shadow_threshold
    lit_bool = ~shadow_bool

    if not np.any(shadow_bool) or not np.any(lit_bool):
        return rgb.astype(np.float32)

    mu_lit = float(np.mean(L_work[lit_bool]))
    mu_sh = float(np.mean(L_work[shadow_bool]))
    L_cap = float(np.percentile(L_work[lit_bool], 92))

    mode = brightness_mode.lower().strip()
    kw, kh = _odd_ksize(mask_blur_blend)
    mask_alpha = cv2.GaussianBlur(m_u8, (kw, kh), 0).astype(np.float32) / 255.0
    if blend_power != 1.0:
        mask_alpha = np.power(np.clip(mask_alpha, 0.0, 1.0), blend_power)

    if mode == "match":
        need = np.clip(mu_lit - L_work, 0.0, None) * brightness_boost
        final_L = np.clip(L_work + need * mask_alpha, 0.0, L_cap)
    else:
        ratio = float(np.clip((mu_lit / (mu_sh + 1e-3)) * brightness_boost, min_ratio, max_ratio))
        factor = 1.0 + mask_alpha * (ratio - 1.0)
        final_L = np.clip(L_work * factor, 0.0, L_cap)

    if verbose:
        print(
            f"      LAB mode={mode} boost={brightness_boost:.2f} "
            f"ratio≈{mu_lit / (mu_sh + 1e-3):.2f} cap_L={L_cap:.1f} "
            f"(lit L≈{mu_lit:.1f}, shadow L≈{mu_sh:.1f})"
        )

    A_f = A.astype(np.float32)
    B_f = Bc.astype(np.float32)
    if match_chroma:
        mu_a_lit = float(np.mean(A_f[lit_bool]))
        mu_b_lit = float(np.mean(B_f[lit_bool]))
        A_f = A_f + (mu_a_lit - A_f) * mask_alpha * chroma_strength
        B_f = B_f + (mu_b_lit - B_f) * mask_alpha * chroma_strength

    final_lab = cv2.merge([
        final_L.astype(np.uint8),
        np.clip(A_f, 0, 255).astype(np.uint8),
        np.clip(B_f, 0, 255).astype(np.uint8),
    ])
    result = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    return (result.astype(np.float32) / 255.0).clip(0.0, 1.0)


# --- Legacy / alternate methods ---


def _soft_mask(mask: np.ndarray, blur_sigma: float = 5.0) -> np.ndarray:
    m = _normalize_mask(mask)
    k = int(max(3, min(31, 2 * int(round(3 * blur_sigma)) + 1)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(m, (k, k), sigmaX=blur_sigma, sigmaY=blur_sigma)


def _lit_shadow_weights(m_soft: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    w_lit = np.clip(1.0 - m_soft, 0.0, 1.0)
    w_sh = np.clip(m_soft, 0.0, 1.0)
    sum_lit = float(np.sum(w_lit) + 1e-6)
    sum_sh = float(np.sum(w_sh) + 1e-6)
    if sum_lit < 500.0:
        thr = float(np.percentile(m_soft, 70))
        w_lit = np.clip(thr - m_soft, 0.0, 1.0)
        sum_lit = float(np.sum(w_lit) + 1e-6)
    return w_lit, w_sh, sum_lit, sum_sh


def _shadow_gain(
    channel: np.ndarray,
    m_soft: np.ndarray,
    *,
    max_gain: float = 2.2,
) -> float:
    w_lit, w_sh, sum_lit, sum_sh = _lit_shadow_weights(m_soft)
    mu_lit = float(np.sum(channel * w_lit) / sum_lit)
    mu_sh = float(np.sum(channel * w_sh) / sum_sh)
    gain = mu_lit / (mu_sh + 1e-3)
    return float(np.clip(gain, 1.0, max_gain))


def remove_shadows_illumination(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    blur_sigma: float = 5.0,
    max_gain: float = 2.2,
) -> np.ndarray:
    """RGB illumination ratio (legacy)."""
    rgb, _ = _prepare_rgb(rgb)
    m_soft = _soft_mask(mask, blur_sigma)
    illum = np.maximum(np.max(rgb, axis=2).astype(np.float32), 1e-3)
    gain = _shadow_gain(illum, m_soft, max_gain=max_gain)
    illum_corr = illum * (1.0 + m_soft * (gain - 1.0))
    scale = np.where(illum > 1e-3, illum_corr / illum, 1.0)
    return np.clip(rgb * scale[..., np.newaxis], 0.0, 1.0).astype(np.float32)


def remove_shadows_hsv(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    blur_sigma: float = 5.0,
    max_gain: float = 2.2,
) -> np.ndarray:
    """HSV value-only brightening (legacy)."""
    rgb, rgb_u8 = _prepare_rgb(rgb)
    m_soft = _soft_mask(mask, blur_sigma)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    gain = _shadow_gain(v, m_soft, max_gain=max_gain)
    v_corr = v * (1.0 + m_soft * (gain - 1.0))
    v_out = ((1.0 - m_soft) * v + m_soft * v_corr).clip(0, 255).astype(np.uint8)
    hsv_out = hsv.copy()
    hsv_out[:, :, 2] = v_out
    rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
    return (rgb_out.astype(np.float32) / 255.0).clip(0.0, 1.0)


def remove_shadows_lab(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """Original affine LAB-L match (legacy; can color-shift)."""
    rgb, rgb_u8 = _prepare_rgb(rgb)
    m_soft = _soft_mask(mask, blur_sigma)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, Bc = cv2.split(lab)
    lf = L.astype(np.float32)
    w_lit, w_sh, sum_lit, sum_sh = _lit_shadow_weights(m_soft)
    mu_lit = np.sum(lf * w_lit) / sum_lit
    var_lit = np.sum(w_lit * (lf - mu_lit) ** 2) / sum_lit
    sig_lit = float(np.sqrt(max(var_lit, 1e-6)))
    mu_sh = np.sum(lf * w_sh) / sum_sh
    var_sh = np.sum(w_sh * (lf - mu_sh) ** 2) / sum_sh
    sig_sh = float(np.sqrt(max(var_sh, 1e-6)))
    scale = sig_lit / (sig_sh + 1e-3)
    lf_corr = (lf - mu_sh) * scale + mu_lit
    lf_out = ((1.0 - m_soft) * lf + m_soft * lf_corr).clip(0, 255).astype(np.uint8)
    lab_out = cv2.merge([lf_out, A, Bc])
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return (rgb_out.astype(np.float32) / 255.0).clip(0.0, 1.0)


def apply_shadow_removal(
    rgb: np.ndarray,
    mask: np.ndarray,
    method: str = "lab-pro",
    **kwargs,
) -> np.ndarray:
    """Dispatch to a CV shadow-removal method."""
    m = method.lower().strip()
    aliases = {"illumination": "lab-pro", "pro": "lab-pro", "professional": "lab-pro"}
    m = aliases.get(m, m)
    if m not in CV_METHODS:
        raise ValueError(f"Unknown cv method {method!r}. Choose from: {', '.join(CV_METHODS)}")
    if m == "lab-pro":
        return remove_shadows_lab_pro(rgb, mask, **kwargs)
    if m == "illumination":
        return remove_shadows_illumination(rgb, mask, **kwargs)
    if m == "hsv":
        return remove_shadows_hsv(rgb, mask, **kwargs)
    return remove_shadows_lab(rgb, mask, **kwargs)
