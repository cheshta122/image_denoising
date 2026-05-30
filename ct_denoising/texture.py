"""Texture region classification for adaptive CT denoising.

Classifies each pixel into one of three regions based on local statistics:
  - homogeneous  : smooth areas (air, uniform soft tissue) → aggressive denoising
  - textured     : complex structure (lungs, trabecular bone) → gentle denoising
  - edge         : boundaries between structures → minimal denoising

The classification drives per-region threshold multipliers so that the
adaptive threshold map respects the underlying tissue characteristics.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# Multipliers applied to the base adaptive threshold map per region.
REGION_MULTIPLIERS = {
    "homogeneous": 1.6,   # Push harder — smooth regions tolerate more shrinkage
    "textured": 0.75,     # Pull back — preserve complex texture signals
    "edge": 0.35,         # Minimal shrinkage — protect structural boundaries
}


def laplacian_variance_map(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Local variance of the Laplacian — a fast focus/texture measure."""
    lap = ndi.laplace(image.astype(np.float64))
    mean = ndi.uniform_filter(lap, size=window_size)
    mean_sq = ndi.uniform_filter(lap * lap, size=window_size)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return _normalize01(variance).astype(np.float32)


def gaussian_local_variance(
    image: np.ndarray, sigma: float = 3.0
) -> np.ndarray:
    """Gaussian-weighted local variance — more spatially coherent than uniform."""
    blurred = ndi.gaussian_filter(image.astype(np.float64), sigma=sigma)
    diff = image.astype(np.float64) - blurred
    variance = ndi.gaussian_filter(diff * diff, sigma=sigma)
    return _normalize01(variance.astype(np.float32))


def classify_regions(
    image: np.ndarray,
    laplacian_sigma: float = 3.0,
    low_percentile: float = 30.0,
    high_percentile: float = 70.0,
) -> np.ndarray:
    """Return an integer region map: 0=homogeneous, 1=textured, 2=edge.

    Uses a mildly pre-smoothed image to reduce noise influence on the
    classification itself — critical when the input is noisy.
    """
    smoothed = ndi.gaussian_filter(image.astype(np.float64), sigma=1.0)
    lv = laplacian_variance_map(smoothed.astype(np.float32))

    low = float(np.percentile(lv, low_percentile))
    high = float(np.percentile(lv, high_percentile))

    region_map = np.ones(image.shape, dtype=np.uint8)  # default: textured
    region_map[lv < low] = 0   # homogeneous
    region_map[lv >= high] = 2  # edge

    return region_map


def texture_multiplier_map(
    image: np.ndarray,
    low_percentile: float = 30.0,
    high_percentile: float = 70.0,
) -> np.ndarray:
    """Build a spatial multiplier map from texture classification.

    Returns a float32 array in [min_multiplier, max_multiplier] that
    can be multiplied element-wise with any threshold map.
    """
    region_map = classify_regions(
        image,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    mult = np.empty(image.shape, dtype=np.float32)
    mult[region_map == 0] = REGION_MULTIPLIERS["homogeneous"]
    mult[region_map == 1] = REGION_MULTIPLIERS["textured"]
    mult[region_map == 2] = REGION_MULTIPLIERS["edge"]

    # Smooth the multiplier boundaries to avoid hard discontinuities
    mult = ndi.gaussian_filter(mult, sigma=2.0).astype(np.float32)

    # Normalize to mean=1.0 so the texture map REDISTRIBUTES the denoising
    # budget rather than globally raising or lowering it.
    # Without this, having many edge pixels pulls the mean below 1.0 and
    # causes systematic under-thresholding across the whole image.
    mean_val = float(np.mean(mult))
    if mean_val > 1e-6:
        mult = mult / mean_val

    return mult


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), copy=False)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)
