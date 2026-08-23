# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Preprocessing math on synthetic arrays; no model files needed."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from retguard.constants import (
    CIRCULAR_MASK_RADIUS_FRACTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
)
from retguard.predictor import _dihedral_views, _oct_views
from retguard.preprocess import (
    _normalize_chw,
    load_image,
    preprocess_dr,
    preprocess_glaucoma,
    preprocess_oct,
)


def _normalized_black() -> np.ndarray:
    """The float32 value each channel takes where the uint8 pixel is 0."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    std = np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    return np.asarray(-mean / std, dtype=np.float32)


def test_fundus_output_shape_and_dtype(synthetic_fundus: np.ndarray) -> None:
    for preprocess in (preprocess_dr, preprocess_glaucoma):
        tensor = preprocess(synthetic_fundus)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert tensor.dtype == np.float32


def test_oct_output_shape_and_dtype(synthetic_oct: np.ndarray) -> None:
    tensor = preprocess_oct(synthetic_oct)
    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_dr_and_glaucoma_share_the_locked_fundus_recipe(
    synthetic_fundus: np.ndarray,
) -> None:
    np.testing.assert_array_equal(
        preprocess_dr(synthetic_fundus), preprocess_glaucoma(synthetic_fundus)
    )


def test_circular_mask_zeroes_corners_at_radius_fraction(
    synthetic_fundus: np.ndarray,
) -> None:
    tensor = preprocess_glaucoma(synthetic_fundus)
    black = _normalized_black()
    for row, col in ((0, 0), (0, INPUT_SIZE - 1), (INPUT_SIZE - 1, 0)):
        np.testing.assert_allclose(tensor[:, row, col], black, atol=1e-6)
    center = INPUT_SIZE // 2
    just_outside = center + int(INPUT_SIZE * CIRCULAR_MASK_RADIUS_FRACTION) + 2
    np.testing.assert_allclose(tensor[:, center, just_outside], black, atol=1e-6)
    assert not np.allclose(tensor[:, center, center], black, atol=1e-6)


def test_enhancement_steps_change_a_plain_resize(synthetic_fundus: np.ndarray) -> None:
    resized = cv2.resize(
        synthetic_fundus, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4
    )
    plain = resized.astype(np.float32)
    plain -= np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    plain /= np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    enhanced = preprocess_glaucoma(synthetic_fundus)
    assert np.abs(enhanced - plain.transpose(2, 0, 1)).max() > 0.1


def test_normalization_matches_imagenet_formula() -> None:
    rgb_values = np.array([[[0, 128, 255]]], dtype=np.uint8)
    tensor = _normalize_chw(rgb_values)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    std = np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    for channel, pixel_value in enumerate((0.0, 128.0, 255.0)):
        expected = (pixel_value - mean[channel]) / std[channel]
        np.testing.assert_allclose(tensor[channel, 0, 0], expected, atol=1e-6)


def test_oct_channels_are_replicated_grayscale(synthetic_oct: np.ndarray) -> None:
    tensor = preprocess_oct(synthetic_oct)
    # Undo the per-channel normalization; the underlying gray plane is shared.
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    std = np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    gray_planes = tensor * std[:, None, None] + mean[:, None, None]
    np.testing.assert_allclose(gray_planes[0], gray_planes[1], atol=1e-3)
    np.testing.assert_allclose(gray_planes[0], gray_planes[2], atol=1e-3)


def test_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError, match="uint8"):
        preprocess_dr(np.zeros((100, 100, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        preprocess_oct(np.zeros((100, 100), dtype=np.uint8))


def test_load_image_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(tmp_path / "absent.jpeg")


def test_load_image_undecodable_file_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "not_an_image.jpeg"
    bad_file.write_text("plain text, not JPEG bytes")
    with pytest.raises(ValueError, match="decode"):
        load_image(bad_file)


def test_load_image_round_trips_rgb(tmp_path: Path, synthetic_oct: np.ndarray) -> None:
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(synthetic_oct, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(load_image(image_path), synthetic_oct)


def test_dihedral_views_are_eight_and_distinct() -> None:
    rng = np.random.default_rng(3)
    tensor = rng.standard_normal((3, 8, 8)).astype(np.float32)
    views = _dihedral_views(tensor)
    assert views.shape == (8, 3, 8, 8)
    np.testing.assert_array_equal(views[0], tensor)
    flattened = {views[index].tobytes() for index in range(8)}
    assert len(flattened) == 8


def test_dihedral_views_invert_to_the_original() -> None:
    rng = np.random.default_rng(4)
    tensor = rng.standard_normal((3, 8, 8)).astype(np.float32)
    views = _dihedral_views(tensor)
    inverses = [
        views[0],
        np.rot90(views[1], k=-1, axes=(1, 2)),
        np.rot90(views[2], k=-2, axes=(1, 2)),
        np.rot90(views[3], k=-3, axes=(1, 2)),
        np.flip(views[4], axis=2),
        np.flip(views[5], axis=1),
        np.flip(np.rot90(views[6], k=-1, axes=(1, 2)), axis=2),
        np.flip(np.rot90(views[7], k=-1, axes=(1, 2)), axis=1),
    ]
    for inverse in inverses:
        np.testing.assert_array_equal(inverse, tensor)


def test_oct_views_are_identity_then_horizontal_flip() -> None:
    rng = np.random.default_rng(5)
    tensor = rng.standard_normal((3, 8, 8)).astype(np.float32)
    views = _oct_views(tensor)
    assert views.shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(views[0], tensor)
    np.testing.assert_array_equal(views[1], np.flip(tensor, axis=2))
