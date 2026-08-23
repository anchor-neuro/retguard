# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Analytic Grad-CAM: head extraction, closed-form gradient, heatmap, exactness."""

import numpy as np
import onnx
import pytest

from retguard.cam import HeadWeights, cam_heatmap, head_logits, pooled_gradient
from retguard.constants import (
    CAM_HEAD_INITIALIZERS,
    CAM_SPATIAL_NODE,
    GAP_OUTPUT_NODE,
    INPUT_SIZE,
    MODULES,
    ONNX_INPUT_NAME,
)
from retguard.predictor import Predictor

_HIDDEN_DIM = 6
_POOLED_DIM = 10


def _random_head(rng: np.random.Generator, num_classes: int) -> HeadWeights:
    return HeadWeights(
        w1=rng.normal(size=(_HIDDEN_DIM, _POOLED_DIM)),
        b1=rng.normal(size=_HIDDEN_DIM),
        w2=rng.normal(size=(num_classes, _HIDDEN_DIM)),
        b2=rng.normal(size=num_classes),
    )


def _minimal_model(rng: np.random.Generator) -> onnx.ModelProto:
    shapes = {
        CAM_HEAD_INITIALIZERS[0]: (_HIDDEN_DIM, _POOLED_DIM),
        CAM_HEAD_INITIALIZERS[1]: (_HIDDEN_DIM,),
        CAM_HEAD_INITIALIZERS[2]: (3, _HIDDEN_DIM),
        CAM_HEAD_INITIALIZERS[3]: (3,),
    }
    initializers = [
        onnx.numpy_helper.from_array(
            rng.normal(size=shape).astype(np.float32), name=name
        )
        for name, shape in shapes.items()
    ]
    graph = onnx.helper.make_graph(
        nodes=[], name="head_only", inputs=[], outputs=[], initializer=initializers
    )
    return onnx.helper.make_model(graph)


class TestHeadWeights:
    def test_from_model_extracts_all_four_initializers(self) -> None:
        rng = np.random.default_rng(0)
        head = HeadWeights.from_model(_minimal_model(rng), "dr")
        assert head.w1.shape == (_HIDDEN_DIM, _POOLED_DIM)
        assert head.b1.shape == (_HIDDEN_DIM,)
        assert head.w2.shape == (3, _HIDDEN_DIM)
        assert head.b2.shape == (3,)

    def test_from_model_rejects_unknown_module(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="module must be one of"):
            HeadWeights.from_model(_minimal_model(rng), "fundus")

    def test_from_model_raises_when_initializer_absent(self) -> None:
        rng = np.random.default_rng(0)
        model = _minimal_model(rng)
        del model.graph.initializer[0]
        with pytest.raises(ValueError, match="not found in the dr graph"):
            HeadWeights.from_model(model, "dr")


class TestPooledGradient:
    @pytest.mark.parametrize("num_classes", [1, 3])
    def test_matches_central_finite_differences(self, num_classes: int) -> None:
        rng = np.random.default_rng(7)
        head = _random_head(rng, num_classes)
        pooled = rng.normal(size=_POOLED_DIM)
        for class_index in range(num_classes):
            gradient = pooled_gradient(pooled, head, class_index)
            epsilon = 1e-6
            for feature_index in range(_POOLED_DIM):
                plus = pooled.copy()
                plus[feature_index] += epsilon
                minus = pooled.copy()
                minus[feature_index] -= epsilon
                finite_difference = (
                    head_logits(plus[None], head)[0, class_index]
                    - head_logits(minus[None], head)[0, class_index]
                ) / (2.0 * epsilon)
                assert abs(gradient[feature_index] - finite_difference) < 1e-6

    def test_class_index_selects_the_w2_row(self) -> None:
        rng = np.random.default_rng(11)
        head = _random_head(rng, 3)
        pooled = rng.normal(size=_POOLED_DIM)
        relu_mask = (pooled @ head.w1.T + head.b1) > 0
        for class_index in range(3):
            expected = (head.w2[class_index] * relu_mask) @ head.w1
            np.testing.assert_allclose(
                pooled_gradient(pooled, head, class_index), expected
            )


class TestCamHeatmap:
    def test_normalizes_to_unit_range_and_upsamples(self) -> None:
        rng = np.random.default_rng(3)
        spatial = rng.normal(size=(_POOLED_DIM, 5, 5)).astype(np.float32)
        gradient = rng.normal(size=_POOLED_DIM)
        heatmap = cam_heatmap(spatial, gradient)
        assert heatmap.shape == (INPUT_SIZE, INPUT_SIZE)
        assert heatmap.dtype == np.float32
        assert float(heatmap.min()) >= 0.0
        assert float(heatmap.max()) <= 1.0

    def test_all_zero_map_stays_all_zero(self) -> None:
        spatial = np.zeros((_POOLED_DIM, 5, 5), dtype=np.float32)
        gradient = np.ones(_POOLED_DIM)
        heatmap = cam_heatmap(spatial, gradient)
        assert not heatmap.any()

    def test_negative_only_response_is_relu_clipped_to_zero(self) -> None:
        spatial = np.ones((_POOLED_DIM, 5, 5), dtype=np.float32)
        gradient = -np.ones(_POOLED_DIM)
        assert not cam_heatmap(spatial, gradient).any()

    def test_constant_positive_map_saturates_inside_the_unit_range(self) -> None:
        """A constant positive map has no spatial signal to min-max normalize;
        it must still honor the [0, 1] contract instead of leaking the raw
        constant (here 2 * _POOLED_DIM)."""
        spatial = np.full((_POOLED_DIM, 5, 5), 2.0, dtype=np.float32)
        gradient = np.ones(_POOLED_DIM)
        heatmap = cam_heatmap(spatial, gradient)
        np.testing.assert_array_equal(heatmap, np.ones_like(heatmap))


@pytest.mark.requires_weights
class TestExactnessOnShippedGraphs:
    """KG-3(a): the three exactness checks on every real artifact."""

    @pytest.fixture(params=MODULES)
    def module_predictor(
        self,
        request: pytest.FixtureRequest,
        dr_predictor: Predictor,
        glaucoma_predictor: Predictor,
        oct_predictor: Predictor,
    ) -> Predictor:
        return {
            "dr": dr_predictor,
            "glaucoma": glaucoma_predictor,
            "oct": oct_predictor,
        }[str(request.param)]

    def _tapped_outputs(
        self, predictor: Predictor, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        tensor = rng.normal(size=(1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        logit, pooled, spatial = predictor._session.run(
            [
                "logit",
                GAP_OUTPUT_NODE[predictor.module],
                CAM_SPATIAL_NODE[predictor.module],
            ],
            {ONNX_INPUT_NAME: tensor},
        )
        return np.atleast_2d(logit), pooled, spatial

    def test_numpy_head_reproduces_graph_logit(self, module_predictor: Predictor) -> None:
        logit, pooled, _ = self._tapped_outputs(module_predictor, seed=5)
        reproduced = head_logits(pooled, module_predictor._cam_head)
        assert np.max(np.abs(reproduced.ravel() - logit.ravel())) < 1e-4

    def test_gap_of_spatial_map_equals_pooled_features(
        self, module_predictor: Predictor
    ) -> None:
        _, pooled, spatial = self._tapped_outputs(module_predictor, seed=6)
        np.testing.assert_allclose(spatial.mean(axis=(2, 3)), pooled, rtol=1e-4, atol=1e-5)

    def test_analytic_gradient_matches_finite_differences(
        self, module_predictor: Predictor
    ) -> None:
        logit, pooled, _ = self._tapped_outputs(module_predictor, seed=7)
        head = module_predictor._cam_head
        pooled64 = pooled[0].astype(np.float64)
        class_index = int(np.argmax(logit[0])) if module_predictor.module == "oct" else 0
        gradient = pooled_gradient(pooled64, head, class_index)
        rng = np.random.default_rng(8)
        epsilon = 1e-4
        for feature_index in rng.choice(pooled64.size, size=20, replace=False):
            plus = pooled64.copy()
            plus[feature_index] += epsilon
            minus = pooled64.copy()
            minus[feature_index] -= epsilon
            finite_difference = (
                head_logits(plus[None], head)[0, class_index]
                - head_logits(minus[None], head)[0, class_index]
            ) / (2.0 * epsilon)
            denominator = max(abs(finite_difference), 1e-8)
            assert abs(gradient[feature_index] - finite_difference) / denominator < 1e-2
