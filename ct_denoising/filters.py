"""Spatial-domain baseline denoising filters.

Improvements:
  - Median filter added (important CT baseline — handles impulse/ring artifacts)
  - Bilateral filter parameters corrected for CT images
    (sigma_color raised from 0.06 → 0.12; spatial raised to 5.0)
  - Gaussian sigma tuned (1.5 is more useful than 1.0 for CT noise levels)
  - NLM (Non-Local Means) added — a strong classical baseline
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma


def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a local mean filter."""
    return np.clip(ndi.uniform_filter(image, size=size), 0.0, 1.0).astype(np.float32)


def gaussian_filter(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Apply Gaussian smoothing.

    sigma=1.5 is more appropriate for typical CT noise levels than 1.0.
    """
    return np.clip(ndi.gaussian_filter(image, sigma=sigma), 0.0, 1.0).astype(np.float32)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a median filter.

    Critical baseline for CT: handles ring artifacts, salt-and-pepper noise,
    and impulse artifacts that are common in low-dose CT reconstruction.
    """
    return np.clip(ndi.median_filter(image, size=size), 0.0, 1.0).astype(np.float32)


def bilateral_filter(
    image: np.ndarray,
    sigma_color: float = 0.12,    # Fixed: was 0.06 (too tight, barely filtered)
    sigma_spatial: float = 5.0,    # Fixed: was 4.0
) -> np.ndarray:
    """Apply edge-preserving bilateral denoising.

    sigma_color=0.12 preserves edges while smoothing noise effectively.
    The original 0.06 was nearly a no-op on CT images.
    """
    filtered = denoise_bilateral(
        image,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    )
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def nlm_filter(image: np.ndarray, patch_size: int = 5, patch_distance: int = 6) -> np.ndarray:
    """Non-Local Means denoising — strong classical baseline.

    Auto-estimates sigma from the image using the MAD estimator so it
    adapts to the actual noise level, not a hardcoded value.
    """
    sigma_est = float(estimate_sigma(image, average_sigmas=True, channel_axis=None))
    h = max(0.6 * sigma_est, 0.01)  # h controls filter strength
    filtered = denoise_nl_means(
        image,
        h=h,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
    )
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def run_baselines(image: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all spatial baseline methods."""
    return {
        "gaussian": gaussian_filter(image),
        "median": median_filter(image),
        "bilateral": bilateral_filter(image),
        "nlm": nlm_filter(image),
    }
