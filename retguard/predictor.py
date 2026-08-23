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
from retguard.cam import HeadWeights, cam_heatmap, pooled_gradient
from retguard.constants import (
    CAM_SPATIAL_NODE,
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
        self._session, self._cam_head = _load_session(module, onnx_path)
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
        tta: bool = True,
    ) -> PredictionResult:
        """Run the module's deployed pipeline on one image.

        Args:
            image: RGB ``(H, W, 3)`` uint8 array, or a path to an image file.
            threshold: Decision threshold. Defaults to the module's
                pre-specified value (dr 0.204983, glaucoma 0.044776,
                oct 0.70; paper section 2.4).
            tta: Average logits over the module's TTA views (8 fundus, 2 OCT;
                paper section 2.3). ``False`` runs the identity view only - a
                deviation from the published protocol, since calibration was
                fitted on TTA-averaged logits. The OOD score uses the identity
                view in both modes.

        Returns:
            The :class:`PredictionResult` for the image.
        """
        result, _ = self._predict_core(image, threshold, tta, want_cam=False)
        return result

    def predict_with_cam(
        self,
        image: np.ndarray | str | Path,
        threshold: float | None = None,
        tta: bool = True,
    ) -> tuple[PredictionResult, np.ndarray]:
        """Run predict and return the exact Grad-CAM heatmap of the identity view.

        Args:
            image: RGB ``(H, W, 3)`` uint8 array, or a path to an image file.
            threshold: Decision threshold. Defaults to the module's
                pre-specified value (dr 0.204983, glaucoma 0.044776,
                oct 0.70; paper section 2.4).
            tta: Average logits over the module's TTA views; see :meth:`predict`.
                The heatmap is identity-view in both modes.

        Returns:
            The PredictionResult and a float32 (INPUT_SIZE, INPUT_SIZE) heatmap in
            [0, 1], computed at the last conv layer via the closed-form head
            gradient and bilinearly upsampled. For oct the map attributes the
            predicted class's logit (pre-softmax).
        """
        result, heatmap = self._predict_core(image, threshold, tta, want_cam=True)
        assert heatmap is not None
        return result, heatmap

    def _predict_core(
        self,
        image: np.ndarray | str | Path,
        threshold: float | None,
        tta: bool,
        want_cam: bool,
    ) -> tuple[PredictionResult, np.ndarray | None]:
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if threshold is None:
            threshold = self.threshold
        tensor = _MODULE_PREPROCESS[self.module](image)
        views = _oct_views(tensor) if self.module == "oct" else _dihedral_views(tensor)
        if not tta:
            views = views[:1]
        logits, features, spatial = self._run_views(views, want_cam=want_cam)
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

        result = PredictionResult(
            module=self.module,
            probability=probability,
            decision=bool(probability >= threshold),
            threshold=threshold,
            class_probabilities=class_probabilities,
            venn_abers_interval=venn_abers_interval,
            ood_score=ood_score,
            ood_flagged=ood_flagged,
        )
        if not want_cam:
            return result, None
        assert spatial is not None
        # Fundus heads are single-logit (class 0); OCT attributes the predicted
        # class's pre-softmax logit, standard Grad-CAM practice (V11_BLUEPRINT §3).
        class_index = (
            int(np.argmax(mean_logit)) if self.module == "oct" else 0
        )
        gradient = pooled_gradient(features[0], self._cam_head, class_index)
        return result, cam_heatmap(spatial[0], gradient)

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

    def _run_views(
        self, views: np.ndarray, want_cam: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Run all TTA views in one batch; return logits, features, spatial map."""
        requested = ["logit", GAP_OUTPUT_NODE[self.module]]
        if want_cam:
            requested.append(CAM_SPATIAL_NODE[self.module])
        outputs = self._session.run(requested, {ONNX_INPUT_NAME: views})
        return outputs[0], outputs[1], outputs[2] if want_cam else None


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


def _load_session(
    module: str, onnx_path: str | Path
) -> tuple[onnxruntime.InferenceSession, HeadWeights]:
    """Build a CPU session with the feature and spatial tensors exposed.

    The shipped graphs output only the head tensors; the OOD gate needs the
    1,280-d post-GAP features and Grad-CAM needs the pre-GAP spatial map.
    Appending the named internal tensors to ``graph.output`` in memory exposes
    them without re-exporting, so the shipped bytes (and their checksums) are
    untouched (REPO_BLUEPRINT section 3.3). The head weights are extracted
    from the same loaded proto - zero extra file reads.
    """
    model = onnx.load(str(onnx_path))
    gap_name = GAP_OUTPUT_NODE[module]
    spatial_name = CAM_SPATIAL_NODE[module]
    produced = {
        tensor_name for node in model.graph.node for tensor_name in node.output
    }
    for tensor_name in (gap_name, spatial_name):
        if tensor_name not in produced:
            raise RuntimeError(
                f"Feature tensor {tensor_name!r} not found in {onnx_path}; the model "
                "file does not match this package version. Re-download with --force."
            )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            gap_name, onnx.TensorProto.FLOAT, ["batch_size", FEATURE_DIM]
        )
    )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            spatial_name, onnx.TensorProto.FLOAT, ["batch_size", FEATURE_DIM, "h", "w"]
        )
    )
    head = HeadWeights.from_model(model, module)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session, head


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
