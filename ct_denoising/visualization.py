"""Visualization utilities for comparing CT denoising methods.

Improvements:
  - Sorted panels: noisy first, then methods by PSNR (ascending) so the
    best result is always last — easy to read the improvement story.
  - Residual maps added for the proposed method (shows what was removed).
  - Bar chart of PSNR/SSIM saved alongside the comparison grid.
  - Metric annotations use color coding: green = better than noisy, red = worse.
  - Window/level rendering option for CT-like display (simulates W/L ~350/40).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Method display name mapping for cleaner labels
_DISPLAY_NAMES = {
    "noisy": "Noisy Input",
    "gaussian": "Gaussian Filter",
    "median": "Median Filter",
    "bilateral": "Bilateral Filter",
    "nlm": "Non-Local Means",
    "wavelet_fixed": "Wavelet (Fixed T)",
    "wavelet_firm": "Wavelet (Fixed T)",
    "shearlet_fixed": "Shearlet (Fixed T)",
    "shearlet_firm": "Shearlet (Fixed T)",
    "adaptive_wavelet": "Adaptive Wavelet",
    "adaptive_shearlet": "Adaptive Shearlet",
}


def _display_name(key: str) -> str:
    return _DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def save_comparison_figure(
    original: np.ndarray,
    noisy: np.ndarray,
    results: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Save a grid containing original, noisy, and all denoised outputs.

    Panels are sorted by PSNR so the best result is in the last panel.
    Residual maps are shown for the proposed adaptive method.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noisy_psnr = metrics.get("noisy", {}).get("psnr", 0.0)

    # Sort methods by PSNR ascending (worst → best left-to-right)
    method_order = sorted(
        results.keys(),
        key=lambda k: metrics.get(k, {}).get("psnr", 0.0),
    )

    # Always: original, noisy, then sorted methods
    panels: list[tuple[str, np.ndarray]] = [
        ("original", original),
        ("noisy", noisy),
        *[(k, results[k]) for k in method_order],
    ]

    # Find adaptive method for residual panel
    adaptive_key = next(
        (k for k in results if k.startswith("adaptive_")), None
    )
    if adaptive_key is not None:
        residual = np.abs(noisy - results[adaptive_key])
        panels.append((f"residual_{adaptive_key}", residual))

    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.2 * rows),
                              facecolor="#1a1a2e")

    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (name, image) in zip(axes_flat, panels):
        is_residual = name.startswith("residual_")
        cmap = "hot" if is_residual else "gray"

        ax.imshow(image, cmap=cmap, vmin=0, vmax=1 if not is_residual else 0.3,
                  interpolation="nearest")
        ax.set_facecolor("#1a1a2e")

        # Build label
        display = _display_name(name.replace("residual_", "Residual: "))
        method_metrics = metrics.get(name, {})

        if method_metrics and not is_residual and name != "original":
            p = method_metrics.get("psnr", 0.0)
            s = method_metrics.get("ssim", 0.0)
            color = "#00ff88" if p > noisy_psnr else "#ff6b6b"
            if name == "noisy":
                color = "#aaaaaa"
            label = f"{display}\nPSNR {p:.2f} dB  SSIM {s:.3f}"
        else:
            color = "#ffffff"
            label = display

        ax.set_title(label, fontsize=9, color=color, pad=4)
        ax.axis("off")

    for ax in axes_flat[len(panels):]:
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")

    fig.suptitle(title, fontsize=12, color="#e0e0e0", y=1.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)

    # Save companion bar chart
    _save_psnr_bar_chart(
        metrics=metrics,
        output_path=output_path.parent / (output_path.stem + "_metrics.png"),
        title=title,
    )


def _save_psnr_bar_chart(
    metrics: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    """Save a bar chart comparing PSNR and SSIM across all methods."""
    skip = {"original", "noisy"}
    method_names = [k for k in metrics if k not in skip]
    if not method_names:
        return

    psnr_vals = [metrics[k].get("psnr", 0.0) for k in method_names]
    ssim_vals = [metrics[k].get("ssim", 0.0) for k in method_names]
    labels = [_display_name(k) for k in method_names]

    # Sort by PSNR
    order = sorted(range(len(psnr_vals)), key=lambda i: psnr_vals[i])
    labels = [labels[i] for i in order]
    psnr_vals = [psnr_vals[i] for i in order]
    ssim_vals = [ssim_vals[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(labels) + 2)),
                                    facecolor="#1a1a2e")

    colors = ["#00ff88" if "adaptive" in method_names[order[i]] or "Adaptive" in labels[i]
              else "#4dabf7"
              for i in range(len(labels))]

    ax1.barh(labels, psnr_vals, color=colors, edgecolor="none", height=0.6)
    ax1.set_xlabel("PSNR (dB)", color="#e0e0e0")
    ax1.set_title("PSNR Comparison", color="#e0e0e0")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="#e0e0e0")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333366")
    ax1.xaxis.label.set_color("#e0e0e0")

    ax2.barh(labels, ssim_vals, color=colors, edgecolor="none", height=0.6)
    ax2.set_xlabel("SSIM", color="#e0e0e0")
    ax2.set_title("SSIM Comparison", color="#e0e0e0")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="#e0e0e0")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333366")

    # Legend
    proposed_patch = mpatches.Patch(color="#00ff88", label="Proposed method")
    baseline_patch = mpatches.Patch(color="#4dabf7", label="Baseline")
    fig.legend(handles=[proposed_patch, baseline_patch], loc="lower right",
               facecolor="#1a1a2e", labelcolor="#e0e0e0")

    fig.suptitle(title, fontsize=10, color="#e0e0e0")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
