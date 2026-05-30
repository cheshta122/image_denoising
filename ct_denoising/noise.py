"""Noise models used to simulate degraded CT slices.

Improvements:
  - Mixed Gaussian+Poisson noise model added — better approximates real
    low-dose CT noise (quantum noise + electronic readout noise).
  - Default Poisson peak lowered to 40 (was 60) — 60 is nearly noiseless;
    40 is more representative of low-dose acquisition.
  - Speckle noise added for completeness.
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.05,
    mean: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise to an image in [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(mean, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_poisson_noise(
    image: np.ndarray,
    peak_photons: float = 40.0,     # Lowered from 60 — more realistic low-dose CT
    seed: int | None = None,
) -> np.ndarray:
    """Simulate signal-dependent Poisson noise.

    The image is interpreted as a normalized photon intensity. Lower
    ``peak_photons`` means stronger quantum noise.
    Clinical low-dose CT: 20–50 photon counts equivalent.
    """
    rng = np.random.default_rng(seed)
    scaled = np.clip(image, 0.0, 1.0) * peak_photons
    noisy = rng.poisson(scaled) / peak_photons
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_mixed_noise(
    image: np.ndarray,
    peak_photons: float = 40.0,
    gaussian_sigma: float = 0.02,   # Electronic readout noise component
    seed: int | None = None,
) -> np.ndarray:
    """Mixed Poisson (quantum) + Gaussian (readout) noise.

    This is the most physically accurate model for CT images:
      - Poisson component: quantum noise proportional to sqrt(signal)
      - Gaussian component: detector electronic readout noise (signal-independent)

    Most clinical CT noise falls into this category.
    """
    # Poisson component
    rng = np.random.default_rng(seed)
    scaled = np.clip(image, 0.0, 1.0) * peak_photons
    poisson_noisy = rng.poisson(scaled) / peak_photons

    # Gaussian readout component
    seed2 = None if seed is None else seed + 1
    gaussian_noisy = add_gaussian_noise(poisson_noisy, sigma=gaussian_sigma, seed=seed2)
    return gaussian_noisy


def add_speckle_noise(
    image: np.ndarray,
    intensity: float = 0.08,
    seed: int | None = None,
) -> np.ndarray:
    """Add multiplicative speckle noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, intensity, size=image.shape)
    noisy = image + image * noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_noise(
    image: np.ndarray,
    noise_type: str,
    seed: int | None = None,
    gaussian_sigma: float = 0.05,
    poisson_peak: float = 40.0,     # Lowered default
) -> np.ndarray:
    """Dispatch supported noise models by name."""
    noise_type = noise_type.lower()
    if noise_type == "gaussian":
        return add_gaussian_noise(image, sigma=gaussian_sigma, seed=seed)
    if noise_type == "poisson":
        return add_poisson_noise(image, peak_photons=poisson_peak, seed=seed)
    if noise_type == "mixed":
        return add_mixed_noise(image, peak_photons=poisson_peak, gaussian_sigma=gaussian_sigma * 0.4, seed=seed)
    if noise_type == "speckle":
        return add_speckle_noise(image, intensity=gaussian_sigma, seed=seed)
    raise ValueError(f"Unsupported noise type: {noise_type!r}")
