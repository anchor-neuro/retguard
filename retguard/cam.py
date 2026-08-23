# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Exact analytic Grad-CAM from the shipped ONNX graphs.

The classifier head is piecewise-linear (GAP -> Gemm -> ReLU -> Gemm), so the
gradient of a class logit with respect to the pooled features has a closed
form, and the gradient through GAP is spatially uniform. The weighted sum of
the pre-GAP spatial map with that gradient is therefore exact Grad-CAM at the
last convolutional layer - not an approximation (research_architecture.md §4).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnx

from retguard.constants import CAM_HEAD_INITIALIZERS, INPUT_SIZE, MODULES


@dataclass(frozen=True)
class HeadWeights:
    """Classifier-head parameters extracted from a shipped ONNX graph.

    Attributes:
        w1: First Gemm weight, shape ``(256, 1280)``.
        b1: First Gemm bias, shape ``(256,)``.
        w2: Second Gemm weight, shape ``(K, 256)`` with K classes.
        b2: Second Gemm bias, shape ``(K,)``.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def from_model(cls, model: onnx.ModelProto, module: str) -> "HeadWeights":
        """Read the head initializers from a loaded module graph.

        Args:
            model: The module's ONNX graph as loaded from the shipped file.
            module: One of ``MODULES``; used only for error context.

        Returns:
            The extracted :class:`HeadWeights`.

        Raises:
            ValueError: If ``module`` is unknown or an initializer is absent.
        """
        if module not in MODULES:
            raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        missing = [name for name in CAM_HEAD_INITIALIZERS if name not in initializers]
        if missing:
            raise ValueError(
                f"Head initializers {missing} not found in the {module} graph; "
                f"the model file does not match this package version. "
                f"Re-download with --force."
            )
        w1, b1, w2, b2 = (
            np.asarray(onnx.numpy_helper.to_array(initializers[name]))
            for name in CAM_HEAD_INITIALIZERS
        )
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)


def head_logits(pooled: np.ndarray, head: HeadWeights) -> np.ndarray:
    """Reproduce the graph's classifier head on pooled features.

    Args:
        pooled: Post-GAP features, shape ``(N, 1280)``.
        head: The module's extracted head parameters.

    Returns:
        Logits of shape ``(N, K)``.
    """
    hidden = pooled @ head.w1.T + head.b1
    logits: np.ndarray = np.maximum(hidden, 0.0) @ head.w2.T + head.b2
    return logits


def pooled_gradient(
    pooled: np.ndarray, head: HeadWeights, class_index: int
) -> np.ndarray:
    """Exact gradient of one class logit with respect to the pooled features.

    Closed form for the piecewise-linear head: ``g = (w2[k] * relu_mask) @ w1``
    with the ReLU mask taken from the actual activation at ``pooled``.

    Args:
        pooled: Post-GAP features of one view, shape ``(1280,)``.
        head: The module's extracted head parameters.
        class_index: Row of ``w2`` to attribute (the class logit).

    Returns:
        Gradient of shape ``(1280,)``.
    """
    relu_mask = (pooled @ head.w1.T + head.b1) > 0
    gradient: np.ndarray = (head.w2[class_index] * relu_mask) @ head.w1
    return gradient


def cam_heatmap(spatial: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmap from a spatial feature map and a channel gradient.

    Args:
        spatial: Pre-GAP feature map of one view, shape ``(1280, h, w)``.
        gradient: Channel gradient from :func:`pooled_gradient`, shape ``(1280,)``.

    Returns:
        Float32 ``(INPUT_SIZE, INPUT_SIZE)`` heatmap in [0, 1], bilinearly
        upsampled; an all-zero map stays all-zero, and a constant positive
        map saturates to ones (it carries no spatial signal to normalize).
    """
    weighted = np.maximum(np.tensordot(gradient, spatial, axes=1), 0.0)
    peak_to_peak = float(weighted.max() - weighted.min())
    if peak_to_peak > 0.0:
        weighted = (weighted - weighted.min()) / peak_to_peak
    elif float(weighted.max()) > 0.0:
        weighted = np.ones_like(weighted)
    resized: np.ndarray = cv2.resize(
        weighted.astype(np.float32),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized
