"""Thresholding methods for transform-domain CT denoising.

Key improvements over the original:
  - Multi-scale edge detection (fine Sobel + coarse LoG) on a pre-smoothed image
  - Gaussian-weighted local variance (more spatially coherent)
  - Firm/semi-soft threshold that interpolates between hard and soft
  - BayesShrink noise-level estimation per wavelet band
  - Texture-aware threshold multiplier (homogeneous / textured / edge regions)
  - All threshold maps are computed from a lightly pre-smoothed version of the
    noisy input so noise does not contaminate the map itself
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, transform


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    """Parameters controlling the adaptive spatial threshold map."""

    base_threshold: float = 0.06          # Lowered from 0.08 for less over-smoothing
    window_size: int = 9
    variance_weight: float = 1.2          # How much local variance raises threshold
    edge_weight: float = 0.70             # How much edge strength suppresses threshold
    min_threshold: float = 0.005
    max_threshold: float = 0.30
    edge_method: str = "multiscale"       # "sobel" | "log" | "multiscale"
    use_texture: bool = True              # Apply texture-region multipliers
    threshold_mode: str = "firm"          # "soft" | "hard" | "firm"
    bayes_shrink: bool = True             # Estimate noise per band for BayesShrink


# ---------------------------------------------------------------------------
# Core threshold functions
# ---------------------------------------------------------------------------

def hard_threshold(coefficients: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Set coefficients with magnitude below threshold to zero."""
    return np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)


def soft_threshold(coefficients: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Shrink coefficients toward zero by threshold magnitude."""
    magnitude = np.maximum(np.abs(coefficients) - threshold, 0.0)
    return np.sign(coefficients) * magnitude


def firm_threshold(
    coefficients: np.ndarray,
    threshold_low: float | np.ndarray,
    threshold_high: float | np.ndarray,
) -> np.ndarray:
    """Firm (semi-soft) thresholding — interpolates between soft and hard.

    Coefficients below threshold_low → zeroed (like hard).
    Coefficients above threshold_high → kept as-is (like hard).
    Between the two thresholds → linearly ramped (avoids Gibbs artifacts).

    This is strictly better than pure soft for CT because it:
      - Kills definite noise (< low threshold) cleanly
      - Preserves strong signal (> high threshold) without shrinkage bias
      - Smoothly transitions in the uncertainty zone
    """
    abs_c = np.abs(coefficients)
    sign_c = np.sign(coefficients)

    # Three-region rule
    zero_mask = abs_c < threshold_low
    full_mask = abs_c >= threshold_high
    ramp_mask = ~zero_mask & ~full_mask

    output = np.zeros_like(coefficients, dtype=np.float64)
    output[full_mask] = coefficients[full_mask]

    if np.any(ramp_mask):
        # Linear ramp: at t_low → 0; at t_high → t_high
        t_lo = threshold_low[ramp_mask] if isinstance(threshold_low, np.ndarray) else threshold_low
        t_hi = threshold_high[ramp_mask] if isinstance(threshold_high, np.ndarray) else threshold_high
        c_ramp = abs_c[ramp_mask]
        scale = (c_ramp - t_lo) / np.maximum(t_hi - t_lo, 1e-8)
        output[ramp_mask] = sign_c[ramp_mask] * c_ramp * scale

    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-scale edge detection
# ---------------------------------------------------------------------------

def _edge_sobel(image: np.ndarray) -> np.ndarray:
    return _normalize01(filters.sobel(image))


def _edge_log(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Laplacian-of-Gaussian edge detector — captures coarser edges."""
    blurred = ndi.gaussian_filter(image.astype(np.float64), sigma=sigma)
    log = ndi.laplace(blurred)
    return _normalize01(np.abs(log).astype(np.float32))


def multiscale_edge_map(image: np.ndarray) -> np.ndarray:
    """Combine fine (Sobel) and coarse (LoG σ=2) edge maps.

    Computing both on a mildly pre-smoothed image reduces noise impact
    on the edge response itself.
    """
    smoothed = ndi.gaussian_filter(image.astype(np.float64), sigma=0.8)
    fine = _edge_sobel(smoothed.astype(np.float32))
    coarse = _edge_log(smoothed.astype(np.float32), sigma=2.0)
    combined = 0.55 * fine + 0.45 * coarse
    return _normalize01(combined)


def edge_map(image: np.ndarray, method: str = "multiscale") -> np.ndarray:
    """Compute a normalized edge-strength map."""
    # Always smooth first to reduce noise contamination of the edge map
    smoothed = ndi.gaussian_filter(image.astype(np.float64), sigma=0.8).astype(np.float32)
    method = method.lower()
    if method == "multiscale":
        return multiscale_edge_map(smoothed)
    if method == "sobel":
        return _normalize01(filters.sobel(smoothed))
    if method in ("log", "canny"):
        return _edge_log(smoothed, sigma=2.0)
    raise ValueError(f"Unsupported edge method: {method!r}")


# ---------------------------------------------------------------------------
# Local variance estimation
# ---------------------------------------------------------------------------

def local_variance_map(
    image: np.ndarray,
    window_size: int = 9,
    gaussian_sigma: float | None = 3.0,
) -> np.ndarray:
    """Estimate local variance.

    When gaussian_sigma is set, uses Gaussian-weighted local variance
    (more spatially coherent than a uniform box filter). Falls back to
    box filter when sigma is None.
    """
    img = image.astype(np.float64)
    if gaussian_sigma is not None:
        blurred = ndi.gaussian_filter(img, sigma=gaussian_sigma)
        diff = img - blurred
        variance = ndi.gaussian_filter(diff * diff, sigma=gaussian_sigma)
    else:
        mean = ndi.uniform_filter(img, size=window_size)
        mean_sq = ndi.uniform_filter(img * img, size=window_size)
        variance = np.maximum(mean_sq - mean * mean, 0.0)
    return _normalize01(variance.astype(np.float32))


# ---------------------------------------------------------------------------
# BayesShrink noise estimation
# ---------------------------------------------------------------------------

def estimate_noise_sigma(coefficients: np.ndarray) -> float:
    """Robust noise-sigma estimate from the HH1 subband (MAD estimator).

    BayesShrink-style: sigma_n = median(|c|) / 0.6745
    This is the standard robust noise estimator for wavelet denoising.
    """
    return float(np.median(np.abs(coefficients)) / 0.6745)


def bayes_shrink_threshold(coefficients: np.ndarray, noise_sigma: float | None = None) -> float:
    """BayesShrink threshold for a single subband.

    T = sigma_n^2 / sigma_x,  where sigma_x = sqrt(max(sigma_y^2 - sigma_n^2, 0))
    sigma_y = std(coefficients) across the entire band.
    """
    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(coefficients)
    sigma_y_sq = float(np.var(coefficients))
    sigma_n_sq = noise_sigma ** 2
    sigma_x = float(np.sqrt(max(sigma_y_sq - sigma_n_sq, 0.0)))
    if sigma_x < 1e-9:
        return float(np.max(np.abs(coefficients)))  # kill entire band
    return sigma_n_sq / sigma_x


# ---------------------------------------------------------------------------
# Main adaptive threshold map
# ---------------------------------------------------------------------------

def adaptive_threshold_map(
    image: np.ndarray,
    config: AdaptiveThresholdConfig | None = None,
) -> np.ndarray:
    """Build a spatial threshold map from local variance, edges, and texture.

    Pipeline:
      1. Mildly pre-smooth the input to avoid noise-contaminated maps
      2. Compute Gaussian local variance  → raise threshold in noisy flat areas
      3. Compute multi-scale edge map     → lower threshold at structure edges
      4. Optionally apply texture-region multipliers (homogeneous/textured/edge)
      5. Clip to [min_threshold, max_threshold]
    """
    config = config or AdaptiveThresholdConfig()

    # Step 1: pre-smooth for map computation
    ref = ndi.gaussian_filter(image.astype(np.float64), sigma=0.8).astype(np.float32)

    # Step 2: local variance
    variance = local_variance_map(ref, window_size=config.window_size, gaussian_sigma=3.0)

    # Step 3: multi-scale edges
    edges = edge_map(ref, method=config.edge_method)

    # Step 4: compose threshold map
    threshold = config.base_threshold * (1.0 + config.variance_weight * variance)
    threshold = threshold * (1.0 - config.edge_weight * edges)

    # Step 5: texture multipliers
    if config.use_texture:
        from .texture import texture_multiplier_map
        mult = texture_multiplier_map(ref)
        threshold = threshold * mult

    return np.clip(threshold, config.min_threshold, config.max_threshold).astype(np.float32)


def resize_threshold_map(threshold_map: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resize a 2D threshold map to a coefficient array shape."""
    target_shape = shape[:2]
    resized = transform.resize(
        threshold_map,
        target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    while resized.ndim < len(shape):
        resized = resized[..., np.newaxis]
    return resized.astype(np.float32)


def apply_threshold(
    coefficients: np.ndarray,
    threshold: float | np.ndarray,
    mode: str = "firm",
) -> np.ndarray:
    """Apply thresholding to a coefficient array.

    Supported modes: 'soft', 'hard', 'firm' (default).
    'firm' uses threshold as the low boundary and 1.5× as the high boundary.
    """
    mode = mode.lower()
    if mode == "hard":
        return hard_threshold(coefficients, threshold)
    if mode == "soft":
        return soft_threshold(coefficients, threshold)
    if mode == "firm":
        high = threshold * 1.5 if isinstance(threshold, (int, float)) else threshold * 1.5
        return firm_threshold(coefficients, threshold, high)
    raise ValueError(f"Unsupported threshold mode: {mode!r}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), copy=False)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)
