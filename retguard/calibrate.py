# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Calibration math: stable activations, temperature scaling, Venn-Abers.

The ``VennAbersCalibrator`` reproduces the released IVAP implementation
(cumulative-sum-diagram convex hulls, equivalent to isotonic regression by
PAVA) and the scalar rule ``p = p1 / (1 - p0 + p1)`` exactly.
"""

from pathlib import Path

import numpy as np

from retguard.constants import (
    VENN_ABERS_DENOMINATOR_FLOOR,
    VENN_ABERS_SLOPE_EPSILON,
)

_VENN_ABERS_NPZ_KEYS = ("cal_scores", "cal_labels", "n_cal", "version")
_VENN_ABERS_VERSION = 1


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic function.

    Args:
        logits: Array of raw logits.

    Returns:
        Array of probabilities in ``[0, 1]`` with the same shape.
    """
    logits = np.asarray(logits, dtype=np.float64)
    bounded: np.ndarray = np.exp(-np.abs(logits))
    return np.asarray(
        np.where(logits >= 0, 1.0 / (1.0 + bounded), bounded / (1.0 + bounded))
    )


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Array of raw logits.
        axis: Axis along which the class dimension runs.

    Returns:
        Array of probabilities summing to 1 along ``axis``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted: np.ndarray = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by a fitted temperature.

    Args:
        logits: Array of raw logits.
        temperature: Positive temperature fitted on a calibration partition.

    Returns:
        Temperature-scaled logits.

    Raises:
        ValueError: If ``temperature`` is not strictly positive.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}. "
            "Use the fitted module temperature from retguard.constants."
        )
    return np.asarray(logits, dtype=np.float64) / temperature


class VennAbersCalibrator:
    """Inductive Venn-Abers Predictor rebuilt from a stored calibration set.

    The shipped ``.npz`` artifact stores the raw calibration score-label pairs;
    the isotonic pair ``(F0, F1)`` is reconstructed deterministically on load,
    exactly as the released ``venn_abers.py`` does.
    """

    def __init__(
        self, cal_scores: np.ndarray, cal_labels: np.ndarray, n_cal: int
    ) -> None:
        self.n_cal = n_cal
        self._pts_unique, y_csd, x_prime, k_prime = _prepare_data(
            cal_scores, cal_labels
        )
        self._F0, self._F1 = _compute_f(x_prime, y_csd, k_prime)

    @classmethod
    def from_npz(cls, path: str | Path) -> "VennAbersCalibrator":
        """Load a calibrator artifact.

        Args:
            path: Path to a ``venn_abers`` ``.npz`` artifact with keys
                ``cal_scores``, ``cal_labels``, ``n_cal``, ``version``.

        Returns:
            A ready calibrator.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If required keys are missing or the artifact version
                is not the supported version 1.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Venn-Abers artifact not found: {path}")
        with np.load(path) as artifact:
            missing_keys = [
                key for key in _VENN_ABERS_NPZ_KEYS if key not in artifact.files
            ]
            if missing_keys:
                raise ValueError(
                    f"Venn-Abers artifact {path} is missing keys {missing_keys}; "
                    f"expected {list(_VENN_ABERS_NPZ_KEYS)}."
                )
            version = int(artifact["version"][0])
            if version != _VENN_ABERS_VERSION:
                raise ValueError(
                    f"Venn-Abers artifact {path} has version {version}, "
                    f"expected {_VENN_ABERS_VERSION}."
                )
            cal_scores = np.asarray(artifact["cal_scores"], dtype=np.float64)
            cal_labels = np.asarray(artifact["cal_labels"], dtype=np.float64)
            n_cal = int(artifact["n_cal"][0])
        return cls(cal_scores, cal_labels, n_cal)

    def calibrate(self, score: float) -> tuple[float, float, float]:
        """Calibrate one uncalibrated probability.

        Args:
            score: Uncalibrated probability in ``[0, 1]`` (the module's
                TTA-averaged sigmoid or softmax output).

        Returns:
            ``(p, p0, p1)`` where ``p = p1 / (1 - p0 + p1)`` is the calibrated
            point probability and ``(p0, p1)`` is the Venn-Abers interval whose
            width is a per-input uncertainty signal.
        """
        scores = np.asarray([score], dtype=np.float64)
        p0, p1 = _get_f_val(self._F0, self._F1, self._pts_unique, scores)
        denominator = np.maximum(1.0 - p0 + p1, VENN_ABERS_DENOMINATOR_FLOOR)
        point = np.clip(p1 / denominator, 0.0, 1.0)
        p0 = np.clip(p0, 0.0, 1.0)
        p1 = np.clip(p1, 0.0, 1.0)
        return float(point[0]), float(p0[0]), float(p1[0])


def _prepare_data(
    cal_scores: np.ndarray, cal_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort calibration pairs and build the cumulative sum diagram."""
    cal_points = sorted(zip(cal_scores.tolist(), cal_labels.tolist()))
    xs = np.array([point[0] for point in cal_points], dtype=np.float64)
    ys = np.array([point[1] for point in cal_points], dtype=np.float64)
    pts_unique, pts_inverse, pts_counts = np.unique(
        xs, return_inverse=True, return_counts=True
    )
    label_sums = np.zeros(pts_unique.shape, dtype=np.float64)
    np.add.at(label_sums, pts_inverse, ys)
    weights = pts_counts.astype(np.float64)
    y_prime = label_sums / weights
    y_csd = np.cumsum(weights * y_prime)
    x_prime = np.cumsum(weights)
    return pts_unique, y_csd, x_prime, len(x_prime)


def _compute_f(
    x_prime: np.ndarray, y_csd: np.ndarray, k_prime: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (F0, F1) slope arrays via CSD convex hulls."""
    csd_points = {0: np.array([0.0, 0.0])}
    for index in range(len(x_prime)):
        csd_points[index + 1] = np.array([x_prime[index], y_csd[index]])
    lower_hull = _lower_hull(csd_points, k_prime)
    F1 = _slopes_from_lower_hull(csd_points, lower_hull, k_prime)

    csd_points[k_prime + 1] = csd_points[k_prime] + np.array([1.0, 0.0])
    upper_hull = _upper_hull(csd_points, k_prime)
    F0 = _slopes_from_upper_hull(csd_points, upper_hull, k_prime)
    return F0, F1


def _get_f_val(
    F0: np.ndarray, F1: np.ndarray, pts_unique: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up F0/F1 values for scores via binary search."""
    pos0 = np.searchsorted(pts_unique, scores, side="left")
    pos1 = np.searchsorted(pts_unique[:-1], scores, side="right") + 1
    pos0 = np.clip(pos0, 0, len(F0) - 1)
    pos1 = np.clip(pos1, 0, len(F1) - 1)
    return F0[pos0], F1[pos1]


def _slope(point_a: np.ndarray, point_b: np.ndarray) -> float:
    delta_x = point_b[0] - point_a[0]
    if abs(delta_x) < VENN_ABERS_SLOPE_EPSILON:
        return float("inf")
    return float((point_b[1] - point_a[1]) / delta_x)


def _cross(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    return float(
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )


def _lower_hull(csd_points: dict[int, np.ndarray], k_prime: int) -> list[int]:
    hull = [0]
    for index in range(1, k_prime + 1):
        while (
            len(hull) > 1
            and _cross(csd_points[hull[-2]], csd_points[hull[-1]], csd_points[index])
            <= 0
        ):
            hull.pop()
        hull.append(index)
    return hull


def _slopes_from_lower_hull(
    csd_points: dict[int, np.ndarray], hull: list[int], k_prime: int
) -> np.ndarray:
    F1 = np.zeros(k_prime + 1)
    position = 0
    for segment in range(len(hull) - 1):
        segment_slope = _slope(csd_points[hull[segment]], csd_points[hull[segment + 1]])
        while position < hull[segment + 1]:
            F1[position + 1] = segment_slope
            position += 1
    return F1


def _upper_hull(csd_points: dict[int, np.ndarray], k_prime: int) -> list[int]:
    hull = [k_prime + 1]
    for index in range(k_prime, -1, -1):
        while (
            len(hull) > 1
            and _cross(csd_points[hull[-2]], csd_points[hull[-1]], csd_points[index])
            >= 0
        ):
            hull.pop()
        hull.append(index)
    hull.reverse()
    return hull


def _slopes_from_upper_hull(
    csd_points: dict[int, np.ndarray], hull: list[int], k_prime: int
) -> np.ndarray:
    F0 = np.zeros(k_prime + 2)
    position = 0
    for segment in range(len(hull) - 1):
        segment_slope = _slope(csd_points[hull[segment]], csd_points[hull[segment + 1]])
        while position < hull[segment + 1]:
            F0[position] = segment_slope
            position += 1
    return F0
