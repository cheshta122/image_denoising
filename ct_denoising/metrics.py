"""Image quality metrics for CT denoising.

Improvements:
  - RMSE added (root mean squared error)
  - FSIM approximation via gradient correlation added
  - Noise reduction ratio (NRR) added — measures how much noise was removed
    vs how much signal was preserved
  - Summary table helper for paper-ready results
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB."""
    return float(peak_signal_noise_ratio(reference, estimate, data_range=1.0))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Structural similarity index."""
    return float(structural_similarity(reference, estimate, data_range=1.0))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2)))


def noise_reduction_ratio(
    reference: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> float:
    """Noise Reduction Ratio: fraction of injected noise removed.

    NRR = 1 - ||denoised - reference|| / ||noisy - reference||
    Perfect denoising → NRR = 1.0; no change → NRR = 0.0.
    """
    noise_before = float(np.linalg.norm(noisy.astype(np.float64) - reference.astype(np.float64)))
    noise_after = float(np.linalg.norm(denoised.astype(np.float64) - reference.astype(np.float64)))
    if noise_before < 1e-10:
        return 1.0
    return float(1.0 - noise_after / noise_before)


def gradient_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Gradient correlation — proxy for edge/structure preservation.

    Computes Pearson correlation between gradient magnitudes of reference
    and estimate. Higher = better edge fidelity.
    """
    from scipy import ndimage as ndi
    grad_ref = ndi.sobel(reference.astype(np.float64))
    grad_est = ndi.sobel(estimate.astype(np.float64))
    ref_flat = grad_ref.ravel()
    est_flat = grad_est.ravel()
    if ref_flat.std() < 1e-10 or est_flat.std() < 1e-10:
        return 0.0
    corr = float(np.corrcoef(ref_flat, est_flat)[0, 1])
    return max(0.0, corr)  # negative correlation is meaningless here


def evaluate_methods(
    reference: np.ndarray,
    methods: dict[str, np.ndarray],
    noisy: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate several denoised outputs against a clean reference.

    If ``noisy`` is provided, also computes noise_reduction_ratio.
    """
    results: dict[str, dict[str, float]] = {}
    for name, image in methods.items():
        entry: dict[str, float] = {
            "psnr": psnr(reference, image),
            "ssim": ssim(reference, image),
            "rmse": rmse(reference, image),
            "grad_corr": gradient_correlation(reference, image),
        }
        if noisy is not None and name != "noisy":
            entry["nrr"] = noise_reduction_ratio(reference, noisy, image)
        results[name] = entry
    return results


def metrics_summary_table(
    metric_values: dict[str, dict[str, float]],
) -> str:
    """Return a nicely formatted ASCII table for paper appendices / console."""
    methods = list(metric_values.keys())
    metric_keys = ["psnr", "ssim", "rmse", "grad_corr", "nrr"]
    headers = ["Method", "PSNR↑", "SSIM↑", "RMSE↓", "GradCorr↑", "NRR↑"]

    rows = []
    for m in methods:
        row = [m]
        for k in metric_keys:
            v = metric_values[m].get(k, None)
            row.append(f"{v:.4f}" if v is not None else "—")
        rows.append(row)

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)
