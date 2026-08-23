# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Per-module inference orchestration: preprocessing, TTA, calibration, OOD."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

from retguard.calibrate import (
    VennAbersCalibrator,
    apply_temperature,
    sigmoid,
    softmax,
)
from retguard.constants import (
    DR_TEMPERATURE,
    DR_THRESHOLD,
    FEATURE_DIM,
    GAP_OUTPUT_NODE,
    GLAUCOMA_THRESHOLD,
    MODULES,
    OCT_AMD_CLASS_INDEX,
    OCT_AMD_THRESHOLD,
    OCT_CLASS_NAMES,
    ONNX_INPUT_NAME,
)
from retguard.ood import MahalanobisGate
from retguard.preprocess import (
    load_image,
    preprocess_dr,
    preprocess_glaucoma,
    preprocess_oct,
)
from retguard.weights import artifact_paths

_MODULE_THRESHOLDS = {
    "dr": DR_THRESHOLD,
    "glaucoma": GLAUCOMA_THRESHOLD,
    "oct": OCT_AMD_THRESHOLD,
}
_MODULE_PREPROCESS = {
    "dr": preprocess_dr,
    "glaucoma": preprocess_glaucoma,
    "oct": preprocess_oct,
}


@dataclass(frozen=True)
class PredictionResult:
    """One image's full inference contract.

    Attributes:
        module: Module identifier (``dr``, ``glaucoma``, or ``oct``).
        probability: Calibrated probability of the target condition -
            referable DR, referable glaucoma, or AMD (calibrated P(AMD)).
        decision: ``probability >= threshold``.
        threshold: The threshold the decision used.
        class_probabilities: OCT only - uncalibrated softmax over
            Normal/AMD/DME from the TTA mean logits; ``None`` for fundus modules.
        venn_abers_interval: ``(p0, p1)`` for the modules with a deployed
            Venn-Abers calibrator (glaucoma, oct); ``None`` for dr. Interval
            width is a per-input uncertainty signal.
        ood_score: Mahalanobis distance of the identity-view features.
        ood_flagged: ``True`` if ``ood_score`` exceeds the module's
            97th-percentile training threshold.
    """

    module: str
    probability: float
    decision: bool
    threshold: float
    class_probabilities: dict[str, float] | None
    venn_abers_interval: tuple[float, float] | None
    ood_score: float
    ood_flagged: bool


class Predictor:
    """Runs one module's deployed pipeline end-to-end on CPU.

    Construct via :func:`load`; the constructor accepts explicit artifact
    paths so tests can point at unpacked artifacts directly.
    """

    def __init__(
        self,
        module: str,
        onnx_path: str | Path,
        ood_path: str | Path,
        venn_abers_path: str | Path | None = None,
    ) -> None:
        if module not in MODULES:
            raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
        if module in ("glaucoma", "oct") and venn_abers_path is None:
            raise ValueError(
                f"module {module!r} requires venn_abers_path: Venn-Abers is its "
                "deployed calibrator (paper section 2.4)."
            )
        self.module = module
        self.threshold = _MODULE_THRESHOLDS[module]
        self._session = _load_session(module, onnx_path)
        self._gate = MahalanobisGate.from_npz(ood_path)
        self._calibrator = (
            VennAbersCalibrator.from_npz(venn_abers_path)
            if venn_abers_path is not None
            else None
        )

    def predict(
        self,
        image: np.ndarray | str | Path,
        threshold: float | None = None,
    ) -> PredictionResult:
        """Run the module's deployed pipeline on one image.

        Args:
            image: RGB ``(H, W, 3)`` uint8 array, or a path to an image file.
            threshold: Decision threshold. Defaults to the module's
                pre-specified value (dr 0.204983, glaucoma 0.044776,
                oct 0.70; paper section 2.4).

        Returns:
            The :class:`PredictionResult` for the image.
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if threshold is None:
            threshold = self.threshold
        tensor = _MODULE_PREPROCESS[self.module](image)
        views = _oct_views(tensor) if self.module == "oct" else _dihedral_views(tensor)
        logits, features = self._run_views(views)
        # TTA contract: mean the raw logits across views, then apply exactly
        # one activation - never average probabilities (tta_utils.py of each
        # module; the code averages logits even where docstrings say otherwise).
        mean_logit = logits.mean(axis=0)
        # OOD uses the identity view's features, matching the released fit and
        # scoring convention (tta_utils.py return_features).
        ood_score = float(self._gate.score(features[:1])[0])
        ood_flagged = bool(ood_score > self._gate.threshold)

        class_probabilities: dict[str, float] | None = None
        venn_abers_interval: tuple[float, float] | None = None
        if self.module == "dr":
            probability = float(sigmoid(apply_temperature(mean_logit, DR_TEMPERATURE)))
        elif self.module == "glaucoma":
            assert self._calibrator is not None
            uncalibrated = float(sigmoid(mean_logit))
            probability, p0, p1 = self._calibrator.calibrate(uncalibrated)
            venn_abers_interval = (p0, p1)
        else:
            assert self._calibrator is not None
            class_vector = softmax(mean_logit)
            class_probabilities = {
                name: float(value)
                for name, value in zip(OCT_CLASS_NAMES, class_vector)
            }
            probability, p0, p1 = self._calibrator.calibrate(
                float(class_vector[OCT_AMD_CLASS_INDEX])
            )
            venn_abers_interval = (p0, p1)

        return PredictionResult(
            module=self.module,
            probability=probability,
            decision=bool(probability >= threshold),
            threshold=threshold,
            class_probabilities=class_probabilities,
            venn_abers_interval=venn_abers_interval,
            ood_score=ood_score,
            ood_flagged=ood_flagged,
        )

    def predict_batch(
        self,
        images: Sequence[np.ndarray | str | Path],
        threshold: float | None = None,
    ) -> list[PredictionResult]:
        """Run :meth:`predict` over a sequence of images.

        Args:
            images: RGB arrays or image paths.
            threshold: Decision threshold applied to every image; defaults to
                the module's pre-specified value.

        Returns:
            One :class:`PredictionResult` per input, in order.
        """
        return [self.predict(image, threshold=threshold) for image in images]

    def _run_views(self, views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run all TTA views in one batch; return per-view logits and features."""
        outputs = self._session.run(
            ["logit", GAP_OUTPUT_NODE[self.module]],
            {ONNX_INPUT_NAME: views},
        )
        return outputs[0], outputs[1]


def load(module: str, weights_dir: str | Path | None = None) -> Predictor:
    """Load one module's predictor from downloaded release artifacts.

    Args:
        module: One of ``MODULES``.
        weights_dir: Directory holding the extracted release members.
            Defaults to ``$RETGUARD_WEIGHTS_DIR`` or ``~/.retguard``.

    Returns:
        A ready :class:`Predictor`.

    Raises:
        ValueError: If ``module`` is not a known module identifier.
        FileNotFoundError: If a required artifact is absent.
    """
    if module not in MODULES:
        raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
    paths = artifact_paths(module, weights_dir)
    for artifact_name, artifact_path in paths.items():
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Model weights ({artifact_name}) not found at {artifact_path}. "
                f"Run 'retguard download --module {module}' (see README)."
            )
    return Predictor(
        module,
        onnx_path=paths["onnx"],
        ood_path=paths["ood"],
        venn_abers_path=paths.get("venn_abers"),
    )


def _load_session(module: str, onnx_path: str | Path) -> onnxruntime.InferenceSession:
    """Build a CPU session with the post-GAP feature tensor exposed.

    The shipped graphs output only the head tensors; the OOD gate needs the
    1,280-d post-GAP features. Appending the named internal tensor to
    ``graph.output`` in memory exposes them without re-exporting, so the
    shipped bytes (and their checksums) are untouched (REPO_BLUEPRINT
    section 3.3).
    """
    model = onnx.load(str(onnx_path))
    gap_name = GAP_OUTPUT_NODE[module]
    produced = {
        tensor_name for node in model.graph.node for tensor_name in node.output
    }
    if gap_name not in produced:
        raise RuntimeError(
            f"Feature tensor {gap_name!r} not found in {onnx_path}; the model "
            "file does not match this package version. Re-download with --force."
        )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            gap_name, onnx.TensorProto.FLOAT, ["batch_size", FEATURE_DIM]
        )
    )
    return onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _dihedral_views(tensor: np.ndarray) -> np.ndarray:
    """Stack the 8 D4 views of a CHW tensor, identity first (tta_utils.py)."""
    horizontal_flip = np.flip(tensor, axis=2)
    vertical_flip = np.flip(tensor, axis=1)
    views = [
        tensor,
        np.rot90(tensor, k=1, axes=(1, 2)),
        np.rot90(tensor, k=2, axes=(1, 2)),
        np.rot90(tensor, k=3, axes=(1, 2)),
        horizontal_flip,
        vertical_flip,
        np.rot90(horizontal_flip, k=1, axes=(1, 2)),
        np.rot90(vertical_flip, k=1, axes=(1, 2)),
    ]
    return np.ascontiguousarray(np.stack(views, axis=0))


def _oct_views(tensor: np.ndarray) -> np.ndarray:
    """Stack the 2 OCT views: identity + horizontal flip (OCT tta_utils.py).

    Rotations and vertical flips are excluded because they invert retinal
    layer ordering.
    """
    return np.ascontiguousarray(
        np.stack([tensor, np.flip(tensor, axis=2)], axis=0)
    )
