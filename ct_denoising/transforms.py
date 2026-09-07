"""Transform-domain denoising backends.

Key improvements over the original:
  - Per-level threshold scaling: finer wavelet levels get HIGHER thresholds
    (contain more noise relative to signal) and coarser levels get LOWER thresholds.
    NOTE: pywt.wavedec2 returns [approx, detail_L, detail_{L-1}, ..., detail_1]
    where index 1 = COARSEST detail, index -1 = FINEST detail.
    We reverse the LEVEL_SCALE_FACTORS so finest = index -1 gets highest multiplier.
  - BayesShrink per-band noise estimation for the fixed threshold path.
  - Firm threshold mode supported end-to-end.
  - Pre-smoothed image used for threshold map computation.
  - db4 wavelet and 4-level decomposition for better frequency separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pywt
from scipy import ndimage as ndi

from .thresholding import (
    AdaptiveThresholdConfig,
    adaptive_threshold_map,
    apply_threshold,
    bayes_shrink_threshold,
    estimate_noise_sigma,
    resize_threshold_map,
)


# Per-level scale factors indexed from COARSEST (index 0) to FINEST (index -1).
# Finer (higher frequency) bands get MORE aggressive thresholding.
# Coarser bands carry structural signal → preserve them more carefully.
COARSE_TO_FINE_SCALES = [0.45, 0.60, 0.80, 1.0, 1.20]


@dataclass
class TransformResult:
    """Denoised image and metadata from a transform-domain method."""

    image: np.ndarray
    backend: str
    threshold_mode: str
    metadata: dict[str, Any]


class TransformDenoiser:
    """Use shearlet denoising when available, otherwise wavelet denoising."""

    def __init__(
        self,
        prefer_shearlet: bool = True,
        wavelet: str = "db4",
        wavelet_level: int = 4,
    ) -> None:
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self._shearlab = None
        if prefer_shearlet:
            self._shearlab = self._try_import_shearlab()

    @property
    def backend_name(self) -> str:
        return "shearlet" if self._shearlab is not None else "wavelet"

    def denoise(
        self,
        noisy_image: np.ndarray,
        threshold: float = 0.06,
        threshold_mode: str = "firm",
        adaptive: bool = False,
        adaptive_config: AdaptiveThresholdConfig | None = None,
    ) -> TransformResult:
        """Denoise an image in transform domain and reconstruct it."""
        if self._shearlab is not None:
            return self._denoise_shearlet(
                noisy_image,
                threshold=threshold,
                threshold_mode=threshold_mode,
                adaptive=adaptive,
                adaptive_config=adaptive_config,
            )
        return self._denoise_wavelet(
            noisy_image,
            threshold=threshold,
            threshold_mode=threshold_mode,
            adaptive=adaptive,
            adaptive_config=adaptive_config,
        )

    def _denoise_wavelet(
        self,
        noisy_image: np.ndarray,
        threshold: float,
        threshold_mode: str,
        adaptive: bool,
        adaptive_config: AdaptiveThresholdConfig | None,
    ) -> TransformResult:
        coeffs = pywt.wavedec2(
            noisy_image,
            wavelet=self.wavelet,
            level=self.wavelet_level,
            mode="periodization",
        )
        approx = coeffs[0]
        detail_levels = coeffs[1:]   # index 0 = coarsest, index -1 = finest

        # Estimate noise sigma from the finest HH band
        hh_finest = detail_levels[-1][2]
        noise_sigma = estimate_noise_sigma(hh_finest)

        # Pre-smoothed reference image for threshold map (reduces noise contamination)
        if adaptive:
            ref = ndi.gaussian_filter(noisy_image.astype(np.float64), sigma=1.0).astype(np.float32)
            thresh_map_2d = adaptive_threshold_map(ref, adaptive_config)

        denoised_coeffs: list[Any] = [approx]
        n_levels = len(detail_levels)

        for level_idx, detail_level in enumerate(detail_levels):
            # level_idx=0: coarsest → scale factor = COARSE_TO_FINE_SCALES[0] (smallest)
            # level_idx=n-1: finest → scale factor = COARSE_TO_FINE_SCALES[-1] (largest)
            scale_idx = min(level_idx, len(COARSE_TO_FINE_SCALES) - 1)
            level_scale = COARSE_TO_FINE_SCALES[scale_idx]

            filtered_bands = []
            for band in detail_level:
                if adaptive:
                    # Spatial adaptive map scaled by level factor
                    band_thresh = resize_threshold_map(
                        (thresh_map_2d * level_scale).astype(np.float32),
                        band.shape,
                    )
                else:
                    # BayesShrink per-band, scaled by level
                    raw_bayes = bayes_shrink_threshold(band, noise_sigma)
                    # BayesShrink can be too aggressive at finest levels — use
                    # a blend: 50% BayesShrink + 50% scaled global threshold
                    blended = 0.5 * raw_bayes + 0.5 * threshold * level_scale
                    band_thresh = float(np.clip(blended, 0.005, 0.30))

                filtered_bands.append(apply_threshold(band, band_thresh, threshold_mode))
            denoised_coeffs.append(tuple(filtered_bands))

        reconstructed = pywt.waverec2(
            denoised_coeffs,
            wavelet=self.wavelet,
            mode="periodization",
        )
        reconstructed = _match_shape(reconstructed, noisy_image.shape)
        return TransformResult(
            image=np.clip(reconstructed, 0.0, 1.0).astype(np.float32),
            backend="wavelet",
            threshold_mode=threshold_mode,
            metadata={
                "adaptive": adaptive,
                "wavelet": self.wavelet,
                "level": self.wavelet_level,
                "noise_sigma_est": float(noise_sigma),
            },
        )

    def _denoise_shearlet(
        self,
        noisy_image: np.ndarray,
        threshold: float,
        threshold_mode: str,
        adaptive: bool,
        adaptive_config: AdaptiveThresholdConfig | None,
    ) -> TransformResult:
        """PyShearLab integration with adaptive threshold map support."""
        shearlab = self._shearlab
        rows, cols = noisy_image.shape
        system = shearlab.SLgetShearletSystem2D(0, rows, cols, 3)
        coeffs = shearlab.SLsheardec2D(noisy_image, system)

        finest_coeffs = coeffs[..., -max(1, coeffs.shape[2] // 4):]
        noise_sigma = estimate_noise_sigma(finest_coeffs)

        if adaptive:
            ref = ndi.gaussian_filter(noisy_image.astype(np.float64), sigma=1.0).astype(np.float32)
            threshold_map = resize_threshold_map(
                adaptive_threshold_map(ref, adaptive_config),
                coeffs.shape,
            )
        else:
            raw_bayes = bayes_shrink_threshold(coeffs[..., 1:], noise_sigma)
            blended = 0.5 * raw_bayes + 0.5 * threshold
            threshold_map = float(np.clip(blended, 0.005, 0.30))

        filtered_coeffs = coeffs.copy()
        threshold_values = threshold_map if isinstance(threshold_map, np.ndarray) else threshold_map
        if isinstance(threshold_values, np.ndarray):
            threshold_values = np.broadcast_to(
                threshold_values,
                filtered_coeffs[..., 1:].shape,
            )
        filtered_coeffs[..., 1:] = apply_threshold(
            filtered_coeffs[..., 1:],
            threshold_values,
            threshold_mode,
        )
        reconstructed = shearlab.SLshearrec2D(filtered_coeffs, system)
        return TransformResult(
            image=np.clip(np.real(reconstructed), 0.0, 1.0).astype(np.float32),
            backend="shearlet",
            threshold_mode=threshold_mode,
            metadata={"adaptive": adaptive, "noise_sigma_est": float(noise_sigma)},
        )

    @staticmethod
    def _try_import_shearlab() -> Any | None:
        try:
            import pyshearlab  # type: ignore
            return pyshearlab
        except Exception:
            vendor_path = Path(__file__).resolve().parents[1] / "vendor" / "pyshearlab"
            if vendor_path.exists() and str(vendor_path) not in sys.path:
                sys.path.insert(0, str(vendor_path))
            try:
                import pyshearlab  # type: ignore
                return pyshearlab
            except Exception:
                return None


def _match_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop reconstructed transform output back to original shape."""
    return image[: shape[0], : shape[1]]
