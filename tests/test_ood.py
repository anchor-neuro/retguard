# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Mahalanobis gate math on a tiny synthetic Gaussian; no model files needed."""

from pathlib import Path

import numpy as np
import pytest

from retguard.constants import FEATURE_DIM, OOD_PERCENTILE
from retguard.ood import MahalanobisGate


def _identity_gate(mean: np.ndarray | None = None) -> MahalanobisGate:
    if mean is None:
        mean = np.zeros(FEATURE_DIM)
    return MahalanobisGate(
        mean=mean,
        cov_cholesky=np.eye(FEATURE_DIM),
        training_scores=np.linspace(0.5, 1.5, 100),
    )


def test_identity_covariance_reduces_to_euclidean_distance() -> None:
    rng = np.random.default_rng(0)
    mean = rng.standard_normal(FEATURE_DIM) * 0.01
    gate = _identity_gate(mean=mean)
    features = rng.standard_normal((5, FEATURE_DIM))
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    expected = np.linalg.norm(normalized - mean, axis=1)
    np.testing.assert_allclose(gate.score(features), expected, atol=1e-9)


def test_score_matches_cholesky_solve_on_a_synthetic_gaussian() -> None:
    rng = np.random.default_rng(1)
    # Small full-rank covariance embedded in the 1,280-d identity.
    covariance = np.eye(FEATURE_DIM)
    block = rng.standard_normal((4, 4))
    covariance[:4, :4] = block @ block.T + 4.0 * np.eye(4)
    cholesky = np.linalg.cholesky(covariance)
    mean = rng.standard_normal(FEATURE_DIM) * 0.01
    gate = MahalanobisGate(
        mean=mean, cov_cholesky=cholesky, training_scores=np.ones(10)
    )
    features = rng.standard_normal((3, FEATURE_DIM))
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    precision = np.linalg.inv(covariance)
    centered = normalized - mean
    expected = np.sqrt(np.einsum("ij,jk,ik->i", centered, precision, centered))
    np.testing.assert_allclose(gate.score(features), expected, atol=1e-9)


def test_scores_are_scale_invariant_via_l2_normalization() -> None:
    rng = np.random.default_rng(2)
    gate = _identity_gate()
    features = rng.standard_normal((2, FEATURE_DIM))
    np.testing.assert_allclose(
        gate.score(features), gate.score(features * 1000.0), atol=1e-9
    )


def test_threshold_is_the_97th_percentile_of_training_scores() -> None:
    training_scores = np.linspace(0.0, 10.0, 1001)
    gate = MahalanobisGate(
        mean=np.zeros(FEATURE_DIM),
        cov_cholesky=np.eye(FEATURE_DIM),
        training_scores=training_scores,
    )
    assert gate.threshold == pytest.approx(
        float(np.percentile(training_scores, OOD_PERCENTILE)), abs=1e-12
    )


def test_flag_compares_score_to_threshold() -> None:
    # Mean equals the L2-normalized feature, so its score is exactly 0.
    gate = _identity_gate(mean=np.eye(FEATURE_DIM)[0])
    feature = np.zeros((1, FEATURE_DIM))
    feature[0, 0] = 2.0
    assert gate.flag(feature, threshold=0.5).tolist() == [False]
    assert gate.flag(feature, threshold=-1.0).tolist() == [True]


def test_wrong_feature_dimension_raises_with_observed_shape() -> None:
    gate = _identity_gate()
    with pytest.raises(ValueError, match=r"\(3, 7\)"):
        gate.score(np.zeros((3, 7)))


def _write_ood_npz(path: Path, drop_key: str | None = None, dim: int = FEATURE_DIM) -> None:
    payload = {
        "mean": np.zeros(dim),
        "cov_cholesky": np.eye(dim),
        "training_scores": np.ones(10),
    }
    if drop_key is not None:
        del payload[drop_key]
    np.savez(path, **payload)


def test_from_npz_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "ood_gate.npz"
    _write_ood_npz(artifact)
    gate = MahalanobisGate.from_npz(artifact)
    assert gate.mean.shape == (FEATURE_DIM,)


def test_from_npz_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        MahalanobisGate.from_npz(tmp_path / "absent.npz")


def test_from_npz_names_missing_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "ood_gate.npz"
    _write_ood_npz(artifact, drop_key="cov_cholesky")
    with pytest.raises(ValueError, match="cov_cholesky"):
        MahalanobisGate.from_npz(artifact)


def test_from_npz_rejects_wrong_shapes(tmp_path: Path) -> None:
    artifact = tmp_path / "ood_gate.npz"
    _write_ood_npz(artifact, dim=8)
    with pytest.raises(ValueError, match="shape"):
        MahalanobisGate.from_npz(artifact)
