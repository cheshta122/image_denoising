"""Run the full CT denoising comparison pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ct_denoising.pipeline import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Scale Edge-Aware CT Image Denoising Experiment."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--noise",
        choices=["gaussian", "poisson", "mixed", "speckle"],
        default="mixed",
        help="Noise model to apply (default: mixed = Poisson + Gaussian readout)"
    )
    parser.add_argument("--gaussian-sigma", type=float, default=0.05)
    parser.add_argument("--poisson-peak", type=float, default=40.0,
                        help="Photon count peak for Poisson noise (lower = noisier)")
    parser.add_argument("--threshold", type=float, default=0.06)
    parser.add_argument(
        "--threshold-mode",
        choices=["hard", "soft", "firm"],
        default="firm",
        help="Threshold mode: firm (recommended) interpolates between hard and soft"
    )
    parser.add_argument(
        "--edge-method",
        choices=["sobel", "log", "multiscale"],
        default="multiscale",
        help="Edge detection strategy (multiscale = Sobel + LoG combined)"
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-shearlet",
        action="store_true",
        help="Force PyWavelets fallback even if PyShearLab is installed.",
    )
    parser.add_argument(
        "--disable-texture",
        action="store_true",
        help="Disable texture-aware threshold multipliers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        input_dir=args.input,
        output_dir=args.output,
        noise_type=args.noise,
        gaussian_sigma=args.gaussian_sigma,
        poisson_peak=args.poisson_peak,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        edge_method=args.edge_method,
        max_images=args.max_images,
        resize=tuple(args.resize) if args.resize else None,
        seed=args.seed,
        prefer_shearlet=not args.disable_shearlet,
        use_texture=not args.disable_texture,
    )
    metrics = run_experiment(config)
    print(f"\nSaved metrics to {config.output_dir / 'metrics.csv'}")
    print(f"Saved figures to {config.output_dir / 'figures/'}")


if __name__ == "__main__":
    main()
