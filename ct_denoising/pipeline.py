"""End-to-end CT denoising experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import ImageSample, demo_phantom, load_grayscale_images, normalize_image
from .filters import run_baselines
from .metrics import evaluate_methods, metrics_summary_table
from .noise import add_noise
from .thresholding import AdaptiveThresholdConfig
from .transforms import TransformDenoiser
from .visualization import save_comparison_figure


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a complete denoising experiment."""

    input_dir: Path
    output_dir: Path
    noise_type: str = "mixed"           # Changed default: mixed is most realistic
    gaussian_sigma: float = 0.05
    poisson_peak: float = 40.0          # Lowered from 60 — more realistic
    threshold: float = 0.06             # Lowered from 0.08
    threshold_mode: str = "firm"        # Changed default to firm
    edge_method: str = "multiscale"     # Changed default to multiscale
    max_images: int | None = None
    resize: tuple[int, int] | None = (256, 256)
    seed: int = 42
    prefer_shearlet: bool = True
    use_texture: bool = True            # Texture-aware thresholding
    use_bayes_shrink: bool = True       # BayesShrink for fixed threshold path


@dataclass(frozen=True)
class SingleImageConfig:
    """Configuration for denoising one CT image."""

    noise_type: str = "mixed"
    gaussian_sigma: float = 0.05
    poisson_peak: float = 40.0
    threshold: float = 0.06
    threshold_mode: str = "firm"
    edge_method: str = "multiscale"
    seed: int = 42
    prefer_shearlet: bool = True
    use_texture: bool = True
    use_bayes_shrink: bool = True


def run_pipeline(
    image: np.ndarray,
    config: SingleImageConfig | None = None,
) -> dict[str, object]:
    """Run the complete denoising pipeline for a single image."""
    config = config or SingleImageConfig()
    original = normalize_image(image)
    transform_denoiser = TransformDenoiser(prefer_shearlet=config.prefer_shearlet)
    adaptive_config = AdaptiveThresholdConfig(
        base_threshold=config.threshold,
        edge_method=config.edge_method,
        use_texture=config.use_texture,
        threshold_mode=config.threshold_mode,
    )

    noisy, results, metric_values, backend = _denoise_arrays(
        original=original,
        config=config,
        transform_denoiser=transform_denoiser,
        adaptive_config=adaptive_config,
        seed=config.seed,
    )

    adaptive_name = f"adaptive_{backend}"
    return {
        "original": original,
        "noisy": noisy,
        "results": results,
        "denoised": results[adaptive_name],
        "adaptive_method": adaptive_name,
        "backend": backend,
        "psnr": metric_values[adaptive_name]["psnr"],
        "ssim": metric_values[adaptive_name]["ssim"],
        "rmse": metric_values[adaptive_name]["rmse"],
        "metrics": metric_values,
        "metrics_table": metrics_summary_table(metric_values),
    }


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run all methods on every input image and save figures plus metrics."""
    samples = load_grayscale_images(
        config.input_dir,
        max_images=config.max_images,
        resize=config.resize,
    )
    if not samples:
        samples = [demo_phantom(size=config.resize[0] if config.resize else 256)]

    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    transform_denoiser = TransformDenoiser(prefer_shearlet=config.prefer_shearlet)
    adaptive_config = AdaptiveThresholdConfig(
        base_threshold=config.threshold,
        edge_method=config.edge_method,
        use_texture=config.use_texture,
        threshold_mode=config.threshold_mode,
    )

    rows: list[dict[str, float | str]] = []
    for index, sample in enumerate(tqdm(samples, desc="Denoising CT images")):
        rows.extend(
            _process_sample(
                sample=sample,
                index=index,
                config=config,
                transform_denoiser=transform_denoiser,
                adaptive_config=adaptive_config,
                figures_dir=figures_dir,
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(config.output_dir / "metrics.csv", index=False)

    # Print summary table
    pivot = metrics_df.groupby("method")[["psnr", "ssim", "rmse"]].mean()
    print("\n=== Method Comparison (averaged across images) ===")
    print(pivot.sort_values("psnr", ascending=False).to_string())

    return metrics_df


def _process_sample(
    sample: ImageSample,
    index: int,
    config: ExperimentConfig,
    transform_denoiser: TransformDenoiser,
    adaptive_config: AdaptiveThresholdConfig,
    figures_dir: Path,
) -> list[dict[str, float | str]]:
    noisy, results, metric_values, backend = _denoise_arrays(
        original=sample.image,
        config=config,
        transform_denoiser=transform_denoiser,
        adaptive_config=adaptive_config,
        seed=config.seed + index,
    )
    save_comparison_figure(
        original=sample.image,
        noisy=noisy,
        results=results,
        metrics=metric_values,
        output_path=figures_dir / f"{sample.name}_comparison.png",
        title=f"{sample.name} | {config.noise_type} noise | {backend} backend",
    )

    rows: list[dict[str, float | str]] = []
    for method, values in metric_values.items():
        rows.append(
            {
                "image": sample.name,
                "method": method,
                "backend": backend if "shearlet" in method or "wavelet" in method else "spatial",
                "noise_type": config.noise_type,
                "psnr": values["psnr"],
                "ssim": values["ssim"],
                "rmse": values["rmse"],
                "grad_corr": values.get("grad_corr", 0.0),
            }
        )
    return rows


def _denoise_arrays(
    original: np.ndarray,
    config: ExperimentConfig | SingleImageConfig,
    transform_denoiser: TransformDenoiser,
    adaptive_config: AdaptiveThresholdConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, float]], str]:
    """Shared implementation for app, single image, and folder experiments."""
    noisy = add_noise(
        original,
        noise_type=config.noise_type,
        seed=seed,
        gaussian_sigma=config.gaussian_sigma,
        poisson_peak=config.poisson_peak,
    )

    results = run_baselines(noisy)

    # Fixed threshold with BayesShrink
    fixed_transform = transform_denoiser.denoise(
        noisy,
        threshold=config.threshold,
        threshold_mode=config.threshold_mode,
        adaptive=False,
    )

    # Adaptive threshold (our method)
    adaptive_transform = transform_denoiser.denoise(
        noisy,
        threshold=config.threshold,
        threshold_mode=config.threshold_mode,
        adaptive=True,
        adaptive_config=adaptive_config,
    )

    backend = adaptive_transform.backend
    results[f"{backend}_fixed"] = fixed_transform.image
    results[f"adaptive_{backend}"] = adaptive_transform.image

    metric_values = evaluate_methods(original, {"noisy": noisy, **results}, noisy=noisy)
    return noisy, results, metric_values, backend
