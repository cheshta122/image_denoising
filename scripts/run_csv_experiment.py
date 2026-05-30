"""Run single-image denoising on paths listed in a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from skimage import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ct_denoising.pipeline import SingleImageConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CT denoising from a CSV file.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with image_path column.")
    parser.add_argument("--output", type=Path, default=Path("outputs/csv_metrics.csv"))
    parser.add_argument("--noise", choices=["gaussian", "poisson"], default="poisson")
    parser.add_argument("--gaussian-sigma", type=float, default=0.05)
    parser.add_argument("--poisson-peak", type=float, default=60.0)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--threshold-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--edge-method", choices=["sobel", "canny"], default="sobel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-shearlet",
        action="store_true",
        help="Force the PyWavelets fallback even if PyShearLab is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain an image_path column.")

    config = SingleImageConfig(
        noise_type=args.noise,
        gaussian_sigma=args.gaussian_sigma,
        poisson_peak=args.poisson_peak,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        edge_method=args.edge_method,
        seed=args.seed,
        prefer_shearlet=not args.disable_shearlet,
    )

    rows = []
    for image_path in df["image_path"]:
        image = io.imread(image_path)
        output = run_pipeline(image, config=config)
        for method, values in output["metrics"].items():
            rows.append(
                {
                    "image_path": image_path,
                    "method": method,
                    "psnr": values["psnr"],
                    "ssim": values["ssim"],
                    "backend": output["backend"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved CSV metrics to {args.output}")


if __name__ == "__main__":
    main()
