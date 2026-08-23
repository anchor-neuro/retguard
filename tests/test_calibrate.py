# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Calibration math: stable activations and the Venn-Abers scalar rule."""

from pathlib import Path

import numpy as np
import pytest

from retguard.calibrate import VennAbersCalibrator, apply_temperature, sigmoid, softmax


def test_sigmoid_matches_definition_in_the_stable_range() -> None:
    logits = np.linspace(-30, 30, 61)
    np.testing.assert_allclose(sigmoid(logits), 1.0 / (1.0 + np.exp(-logits)), atol=1e-12)


def test_sigmoid_is_stable_at_extreme_logits() -> None:
    values = sigmoid(np.array([-500.0, 500.0]))
    assert values[0] == pytest.approx(0.0, abs=1e-100)
    assert values[1] == pytest.approx(1.0, abs=1e-100)
    assert np.all(np.isfinite(values))


def test_softmax_sums_to_one_and_is_stable_at_extreme_logits() -> None:
    logits = np.array([[500.0, -500.0, 0.0], [3.0, 1.0, -2.0]])
    probabilities = softmax(logits)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12)
    assert probabilities[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_apply_temperature_divides_logits() -> None:
    logits = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(
        apply_temperature(logits, 0.841517), logits / 0.841517, atol=1e-12
    )


def test_apply_temperature_rejects_nonpositive_values() -> None:
    for bad_temperature in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(np.array([1.0]), bad_temperature)


def _separable_calibrator() -> VennAbersCalibrator:
    # Perfectly separated calibration set: label 0 below 0.5, label 1 above.
    cal_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    cal_labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return VennAbersCalibrator(cal_scores, cal_labels, n_cal=len(cal_scores))


def test_scalar_rule_algebra_on_hand_computed_intervals() -> None:
    # Hand-computed from the cumulative-sum-diagram hulls of the separable set:
    # below every calibration point both hull slopes are 0, so p = 0; above,
    # (p0, p1) = (0.75, 1.0) and p = 1.0 / (1 - 0.75 + 1.0) = 0.8.
    calibrator = _separable_calibrator()
    low, low_p0, low_p1 = calibrator.calibrate(0.05)
    assert (low, low_p0, low_p1) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    high, high_p0, high_p1 = calibrator.calibrate(0.95)
    assert (high_p0, high_p1) == pytest.approx((0.75, 1.0), abs=1e-12)
    assert high == pytest.approx(high_p1 / (1.0 - high_p0 + high_p1), abs=1e-12)
    assert high == pytest.approx(0.8, abs=1e-12)


def test_calibrated_probability_is_monotone_in_the_score() -> None:
    calibrator = _separable_calibrator()
    points = [calibrator.calibrate(score)[0] for score in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert points == sorted(points)
    assert points[0] < 0.5 < points[-1]


def test_interval_brackets_the_point_probability() -> None:
    calibrator = _separable_calibrator()
    for score in (0.05, 0.4, 0.6, 0.95):
        calibrated, p0, p1 = calibrator.calibrate(score)
        assert 0.0 <= p0 <= p1 <= 1.0
        assert 0.0 <= calibrated <= 1.0


def _write_venn_abers_npz(path: Path, drop_key: str | None = None, version: int = 1) -> None:
    payload = {
        "cal_scores": np.array([0.1, 0.9]),
        "cal_labels": np.array([0.0, 1.0]),
        "n_cal": np.array([2]),
        "version": np.array([version]),
    }
    if drop_key is not None:
        del payload[drop_key]
    np.savez(path, **payload)


def test_from_npz_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "venn_abers.npz"
    _write_venn_abers_npz(artifact)
    calibrator = VennAbersCalibrator.from_npz(artifact)
    assert calibrator.n_cal == 2


def test_from_npz_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        VennAbersCalibrator.from_npz(tmp_path / "absent.npz")


def test_from_npz_names_missing_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "venn_abers.npz"
    _write_venn_abers_npz(artifact, drop_key="cal_labels")
    with pytest.raises(ValueError, match="cal_labels"):
        VennAbersCalibrator.from_npz(artifact)


def test_from_npz_rejects_unknown_version(tmp_path: Path) -> None:
    artifact = tmp_path / "venn_abers.npz"
    _write_venn_abers_npz(artifact, version=2)
    with pytest.raises(ValueError, match="version"):
        VennAbersCalibrator.from_npz(artifact)
