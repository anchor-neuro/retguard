# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Shared fixtures: synthetic images and weights-dependent predictors."""

from pathlib import Path

import numpy as np
import pytest

from retguard.constants import MODULES
from retguard.predictor import Predictor, load
from retguard.weights import artifact_paths, default_weights_dir


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    """Directory with extracted release artifacts, or skip if absent."""
    base = default_weights_dir()
    missing = [
        str(path)
        for module in MODULES
        for path in artifact_paths(module, base).values()
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            f"release artifacts absent under {base} (set RETGUARD_WEIGHTS_DIR "
            f"or run 'retguard download'); missing: {missing[:3]}"
        )
    return base


@pytest.fixture(scope="session")
def dr_predictor(weights_dir: Path) -> Predictor:
    return load("dr", weights_dir=weights_dir)


@pytest.fixture(scope="session")
def glaucoma_predictor(weights_dir: Path) -> Predictor:
    return load("glaucoma", weights_dir=weights_dir)


@pytest.fixture(scope="session")
def oct_predictor(weights_dir: Path) -> Predictor:
    return load("oct", weights_dir=weights_dir)


def make_synthetic_fundus(seed: int, height: int = 620, width: int = 700) -> np.ndarray:
    """Fundus-like RGB uint8 image: bright textured disc on a black frame."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    center_y, center_x = height // 2, width // 2
    radius = int(min(height, width) * 0.45)
    disc = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2
    texture = rng.integers(40, 220, size=(height, width, 3), dtype=np.uint8)
    gradient = (60 + 100 * xx / width + 40 * yy / height).astype(np.uint8)[..., None]
    image[disc] = (0.6 * texture[disc] + 0.4 * gradient.repeat(3, axis=2)[disc]).astype(np.uint8)
    return image


def make_synthetic_oct(seed: int, height: int = 500, width: int = 760) -> np.ndarray:
    """OCT-like RGB uint8 image: dark borders and horizontal layer bands."""
    rng = np.random.default_rng(seed)
    gray = np.zeros((height, width), dtype=np.uint8)
    band_top = int(height * 0.3)
    band_bottom = int(height * 0.75)
    yy = np.arange(height)[:, None]
    bands = 120 + 80 * np.sin((yy - band_top) / 9.0) + rng.normal(0, 18, size=(height, width))
    inside = np.broadcast_to((yy >= band_top) & (yy < band_bottom), gray.shape)
    gray[inside] = np.clip(bands, 0, 255).astype(np.uint8)[inside]
    pad = int(width * 0.08)
    gray[:, :pad] = 0
    gray[:, -pad:] = 0
    return np.stack([gray, gray, gray], axis=2)


@pytest.fixture()
def synthetic_fundus() -> np.ndarray:
    return make_synthetic_fundus(seed=7)


@pytest.fixture()
def synthetic_oct() -> np.ndarray:
    return make_synthetic_oct(seed=11)
