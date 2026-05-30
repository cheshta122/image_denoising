"""Image input utilities for CT denoising experiments.

Improvements:
  - Richer demo phantom with rings, Gaussian blobs, and noise texture
    to simulate real CT tissue heterogeneity.
  - DICOM loading support (requires pydicom, graceful fallback).
  - Contrast stretching option for better visual range utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import color, data, img_as_float32, io, transform


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}


@dataclass(frozen=True)
class ImageSample:
    """A normalized grayscale image with source metadata."""

    name: str
    image: np.ndarray
    path: Path | None = None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in [0, 1] with full contrast stretch."""
    image = img_as_float32(image)
    if image.ndim == 3:
        image = color.rgb2gray(image[..., :3])

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _load_dicom(path: Path) -> np.ndarray | None:
    """Attempt to load a DICOM file. Returns None if pydicom not available."""
    try:
        import pydicom  # type: ignore
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)
        return pixel_array
    except Exception:
        return None


def load_grayscale_images(
    folder: str | Path,
    max_images: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[ImageSample]:
    """Load grayscale images from a folder, sorted by filename."""
    folder = Path(folder)
    if not folder.exists():
        return []

    paths = sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]

    samples: list[ImageSample] = []
    for path in paths:
        if path.suffix.lower() == ".dcm":
            raw = _load_dicom(path)
            if raw is None:
                continue
            image = normalize_image(raw)
        else:
            image = normalize_image(io.imread(path))

        if resize is not None:
            image = transform.resize(
                image,
                resize,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
        samples.append(ImageSample(name=path.stem, image=image, path=path))
    return samples


def demo_phantom(size: int = 256) -> ImageSample:
    """Create a rich CT-like phantom when no external CT slices are available.

    Layers:
      - Shepp-Logan base (ellipsoidal structures like a head CT)
      - Gaussian blobs simulating soft-tissue inhomogeneities
      - Ring pattern simulating bone cortex / trabecular transition
      - Mild background texture
    """
    # Base Shepp-Logan
    phantom = data.shepp_logan_phantom()
    phantom = transform.resize(
        phantom,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float64)

    # Add structured Gaussian blobs to simulate tissue heterogeneity
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:size, 0:size]
    for _ in range(8):
        cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
        sigma_blob = rng.uniform(size // 30, size // 15)
        amplitude = rng.uniform(0.04, 0.12)
        blob = amplitude * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma_blob ** 2))
        phantom += blob

    # Add concentric ring pattern to simulate bone structure
    center = size // 2
    radius = np.sqrt((y - center) ** 2 + (x - center) ** 2)
    ring_pattern = 0.03 * np.sin(radius / (size / 20) * np.pi)
    phantom += ring_pattern

    # Mild background texture
    texture_noise = 0.01 * rng.standard_normal((size, size))
    phantom += texture_noise

    return ImageSample(
        name="demo_ct_phantom",
        image=normalize_image(phantom.astype(np.float32)),
        path=None,
    )
