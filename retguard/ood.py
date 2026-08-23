# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Mahalanobis out-of-distribution gate on 1,280-d post-GAP features."""

from pathlib import Path

import numpy as np

from retguard.constants import FEATURE_DIM, FEATURE_NORM_FLOOR, OOD_PERCENTILE

_OOD_NPZ_KEYS = ("mean", "cov_cholesky", "training_scores")


class MahalanobisGate:
    """Pooled-Gaussian Mahalanobis distance gate (Mahalanobis++ variant).

    Feature rows are L2-normalized, then scored as the Mahalanobis distance
    (not its square) to the pooled training mean, computed by forward
    substitution against the stored Cholesky factor of the ridge-regularized
    empirical covariance. Higher scores are more out-of-distribution.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov_cholesky: np.ndarray,
        training_scores: np.ndarray,
    ) -> None:
        self.mean = mean
        self.cov_cholesky = cov_cholesky
        self.training_scores = training_scores
        # Deployed flag threshold: the 97th percentile of training scores,
        # recomputed from the artifact exactly as the released ood_gate.py does.
        self.threshold = float(np.percentile(training_scores, OOD_PERCENTILE))

    @classmethod
    def from_npz(cls, path: str | Path) -> "MahalanobisGate":
        """Load a fitted gate artifact.

        Args:
            path: Path to an ``ood_gate`` ``.npz`` artifact with keys ``mean``,
                ``cov_cholesky``, ``training_scores``.

        Returns:
            A ready gate.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If required keys are missing or shapes are not
                ``(1280,)`` / ``(1280, 1280)``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"OOD gate artifact not found: {path}")
        with np.load(path) as artifact:
            missing_keys = [key for key in _OOD_NPZ_KEYS if key not in artifact.files]
            if missing_keys:
                raise ValueError(
                    f"OOD gate artifact {path} is missing keys {missing_keys}; "
                    f"expected {list(_OOD_NPZ_KEYS)}."
                )
            mean = artifact["mean"]
            cov_cholesky = artifact["cov_cholesky"]
            training_scores = artifact["training_scores"]
        if mean.shape != (FEATURE_DIM,) or cov_cholesky.shape != (
            FEATURE_DIM,
            FEATURE_DIM,
        ):
            raise ValueError(
                f"OOD gate artifact {path} has mean shape {mean.shape} and "
                f"cov_cholesky shape {cov_cholesky.shape}; expected "
                f"({FEATURE_DIM},) and ({FEATURE_DIM}, {FEATURE_DIM})."
            )
        return cls(mean, cov_cholesky, training_scores)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score feature rows by Mahalanobis distance.

        Args:
            features: ``(N, 1280)`` array of post-GAP backbone features
                (un-normalized; L2 normalization is applied here).

        Returns:
            ``(N,)`` array of distances; higher means more out-of-distribution.

        Raises:
            ValueError: If ``features`` is not a 2-D array with 1,280 columns.
        """
        features = np.atleast_2d(features)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected (N, {FEATURE_DIM}) features, got shape {features.shape}."
            )
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.maximum(norms, FEATURE_NORM_FLOOR)
        centered = normalized - self.mean
        solved = np.linalg.solve(self.cov_cholesky, centered.T)
        mahalanobis_squared = np.sum(solved**2, axis=0)
        return np.sqrt(np.maximum(mahalanobis_squared, 0.0))

    def flag(self, features: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Flag out-of-distribution feature rows.

        Args:
            features: ``(N, 1280)`` array of post-GAP backbone features.
            threshold: Flag threshold; defaults to the artifact's 97th-percentile
                training-score threshold (paper section 2.4).

        Returns:
            ``(N,)`` boolean array; ``True`` means flagged as out-of-distribution.
        """
        if threshold is None:
            threshold = self.threshold
        return self.score(features) > threshold
