# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Image loading and per-module preprocessing.

Each ``preprocess_*`` function reproduces the released training-domain recipe
byte-for-byte on the uint8 stage, then applies ImageNet normalization and
returns the CHW float32 tensor the ONNX models consume.
"""

from pathlib import Path

import cv2
import numpy as np

from retguard.constants import (
    BEN_GRAHAM_GAIN,
    BEN_GRAHAM_OFFSET,
    BEN_GRAHAM_SIGMA_DIVISOR,
    CIRCULAR_MASK_RADIUS_FRACTION,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID,
    FUNDUS_CROP_INTENSITY_THRESHOLD,
    FUNDUS_CROP_MIN_BOX_FRACTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    OCT_BORDER_CROP_MIN_ROI_FRACTION,
    OCT_BORDER_CROP_THRESHOLD,
    OCT_PERCENTILE_HIGH,
    OCT_PERCENTILE_LOW,
    OCT_PERCENTILE_MIN_RANGE,
)


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image file as an RGB array.

    Args:
        image_path: Path to a JPEG or PNG image.

    Returns:
        Image as an ``(H, W, 3)`` uint8 RGB array.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(
            f"Cannot decode {image_path} as an image. Supply a readable JPEG or PNG file."
        )
    return np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def preprocess_dr(image: np.ndarray) -> np.ndarray:
    """Preprocess a fundus photograph for the DR module.

    The DR source recipe (preprocess.py, "LOCKED") is identical to the glaucoma
    recipe: circle crop, Lanczos4 resize to 480, Ben Graham unsharp masking,
    CLAHE on the LAB L-channel, circular mask, then ImageNet normalization.

    Args:
        image: ``(H, W, 3)`` uint8 RGB fundus photograph.

    Returns:
        ``(3, 480, 480)`` float32 CHW tensor, ImageNet-normalized.
    """
    return _normalize_chw(_preprocess_fundus(image))


def preprocess_glaucoma(image: np.ndarray) -> np.ndarray:
    """Preprocess a fundus photograph for the glaucoma module.

    Five uint8 steps in the artifact order (glaucoma preprocess.py): circle
    crop, Lanczos4 resize to 480, Ben Graham unsharp (``4*img - 4*blur + 128``,
    sigma = size/30), CLAHE (clip 2.0, 8x8 grid) on the LAB L-channel, circular
    mask at radius ``0.45 * size`` - then ImageNet normalization.

    Args:
        image: ``(H, W, 3)`` uint8 RGB fundus photograph.

    Returns:
        ``(3, 480, 480)`` float32 CHW tensor, ImageNet-normalized.
    """
    return _normalize_chw(_preprocess_fundus(image))


def preprocess_oct(image: np.ndarray) -> np.ndarray:
    """Preprocess an OCT B-scan for the OCT module.

    Reproduces preprocess_oct.py: border crop, Lanczos4 resize to 480,
    grayscale conversion, percentile clip to [1st, 99th] rescaled to [0, 255],
    CLAHE (clip 2.0, 8x8 grid), replication to three channels, then ImageNet
    normalization.

    Args:
        image: ``(H, W, 3)`` uint8 RGB B-scan (grayscale content is typical).

    Returns:
        ``(3, 480, 480)`` float32 CHW tensor, ImageNet-normalized.
    """
    _validate_uint8_rgb(image)
    cropped = _border_crop(image)
    resized = cv2.resize(
        cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = _percentile_normalize(gray)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID)
    gray = clahe.apply(gray)
    return _normalize_chw(cv2.merge([gray, gray, gray]))


def _validate_uint8_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected an (H, W, 3) uint8 RGB array, got shape {image.shape} "
            f"dtype {image.dtype}. Use retguard.preprocess.load_image to load files."
        )


def _preprocess_fundus(image: np.ndarray) -> np.ndarray:
    """Shared uint8 fundus pipeline; operates in BGR to match the cv2 source."""
    _validate_uint8_rgb(image)
    # The released pipeline runs on cv2.imread output (BGR). Convert, run the
    # recipe in BGR, and convert back so LAB and grayscale math is bit-identical.
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_image = _circle_crop(bgr_image)
    bgr_image = cv2.resize(
        bgr_image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4
    )
    bgr_image = _ben_graham(bgr_image)
    bgr_image = _clahe_lab(bgr_image)
    bgr_image = _circular_mask(bgr_image)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def _circle_crop(bgr_image: np.ndarray) -> np.ndarray:
    """Crop to the fundus disc's bounding box when it spans enough of the frame."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, FUNDUS_CROP_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(largest_contour)
        if (
            box_w > bgr_image.shape[1] * FUNDUS_CROP_MIN_BOX_FRACTION
            and box_h > bgr_image.shape[0] * FUNDUS_CROP_MIN_BOX_FRACTION
        ):
            bgr_image = bgr_image[y : y + box_h, x : x + box_w]
    return bgr_image


def _ben_graham(bgr_image: np.ndarray) -> np.ndarray:
    """Ben Graham unsharp masking: 4*img - 4*blur + 128, sigma = size/30."""
    sigma = INPUT_SIZE / BEN_GRAHAM_SIGMA_DIVISOR
    blurred = cv2.GaussianBlur(bgr_image, (0, 0), sigmaX=sigma)
    enhanced = cv2.addWeighted(
        bgr_image, BEN_GRAHAM_GAIN, blurred, -BEN_GRAHAM_GAIN, BEN_GRAHAM_OFFSET
    )
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _clahe_lab(bgr_image: np.ndarray) -> np.ndarray:
    """CLAHE on the L-channel in LAB space."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _circular_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Zero everything outside the centered circle of radius 0.45 * size."""
    mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
    radius = int(INPUT_SIZE * CIRCULAR_MASK_RADIUS_FRACTION)
    cv2.circle(mask, (INPUT_SIZE // 2, INPUT_SIZE // 2), radius, 255, -1)
    return cv2.bitwise_and(bgr_image, bgr_image, mask=mask)


def _border_crop(image: np.ndarray) -> np.ndarray:
    """Crop black scanner padding when the bright ROI is large enough."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright_pixels = cv2.findNonZero(
        (gray > OCT_BORDER_CROP_THRESHOLD).astype(np.uint8)
    )
    if bright_pixels is None:
        return image
    x, y, roi_w, roi_h = cv2.boundingRect(bright_pixels)
    height, width = image.shape[:2]
    if (
        roi_w > width * OCT_BORDER_CROP_MIN_ROI_FRACTION
        and roi_h > height * OCT_BORDER_CROP_MIN_ROI_FRACTION
    ):
        return image[y : y + roi_h, x : x + roi_w]
    return image


def _percentile_normalize(gray: np.ndarray) -> np.ndarray:
    """Clip to the [1st, 99th] percentile band and rescale to [0, 255]."""
    values = gray.astype(np.float32)
    p_low = np.percentile(values, OCT_PERCENTILE_LOW)
    p_high = np.percentile(values, OCT_PERCENTILE_HIGH)
    if p_high - p_low < OCT_PERCENTILE_MIN_RANGE:
        return gray
    values = np.clip(values, p_low, p_high)
    values = (values - p_low) / (p_high - p_low) * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _normalize_chw(rgb_image: np.ndarray) -> np.ndarray:
    """ImageNet-normalize a uint8 RGB image and transpose to CHW float32."""
    # Matches albumentations Normalize(max_pixel_value=255): subtract mean*255,
    # multiply by the reciprocal of std*255, all in float32.
    normalized = rgb_image.astype(np.float32)
    normalized -= np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    normalized *= np.float32(1.0) / (np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))
