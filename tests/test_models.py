# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Model-execution tests against the real ONNX release artifacts."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import make_synthetic_fundus, make_synthetic_oct

from retguard import cli
from retguard.calibrate import sigmoid, softmax
from retguard.constants import FEATURE_DIM, MODULE_OUTPUT_NAMES, ONNX_INPUT_NAME
from retguard.predictor import Predictor, _load_session
from retguard.preprocess import preprocess_dr, preprocess_oct
from retguard.weights import artifact_paths

pytestmark = pytest.mark.requires_weights


def _predictor(module: str, request: pytest.FixtureRequest) -> Predictor:
    fixture: Predictor = request.getfixturevalue(f"{module}_predictor")
    return fixture


def _input_batch(module: str, count: int = 2) -> np.ndarray:
    if module == "oct":
        tensors = [preprocess_oct(make_synthetic_oct(seed)) for seed in range(count)]
    else:
        tensors = [preprocess_dr(make_synthetic_fundus(seed)) for seed in range(count)]
    return np.stack(tensors, axis=0)


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_interface_conformance(module: str, request: pytest.FixtureRequest) -> None:
    predictor = _predictor(module, request)
    batch = _input_batch(module)
    outputs = predictor._session.run(
        list(MODULE_OUTPUT_NAMES[module]), {ONNX_INPUT_NAME: batch}
    )
    named = dict(zip(MODULE_OUTPUT_NAMES[module], outputs))
    if module == "oct":
        assert named["probability"].shape == (2, 3)
        assert named["logit"].shape == (2, 3)
        assert named["p_amd"].shape == (2,)
    else:
        assert named["probability"].shape == (2,)
        assert named["logit"].shape == (2,)
    for tensor in outputs:
        assert tensor.dtype == np.float32
        assert np.all(np.isfinite(tensor))


@pytest.mark.parametrize("module", ["dr", "glaucoma"])
def test_fundus_probability_output_is_sigmoid_of_logit(
    module: str, request: pytest.FixtureRequest
) -> None:
    # This bounds in-graph sigmoid(logit)-vs-probability consistency (measured
    # ~1e-8 for both fundus graphs). The DR export's documented logit-parity
    # failure (0.002526 vs 0.001; DR model card section 10) is a PyTorch-vs-ONNX
    # diff, a different quantity, and does not loosen this bound.
    predictor = _predictor(module, request)
    batch = _input_batch(module)
    probability, logit = predictor._session.run(
        ["probability", "logit"], {ONNX_INPUT_NAME: batch}
    )
    observed = float(np.max(np.abs(sigmoid(logit) - probability)))
    assert observed < 1e-4


def test_oct_probability_output_is_softmax_of_logit(
    request: pytest.FixtureRequest,
) -> None:
    predictor = _predictor("oct", request)
    batch = _input_batch("oct")
    probability, logit = predictor._session.run(
        ["probability", "logit"], {ONNX_INPUT_NAME: batch}
    )
    assert float(np.max(np.abs(softmax(logit) - probability))) < 1e-4


def test_oct_p_amd_equals_probability_column(request: pytest.FixtureRequest) -> None:
    predictor = _predictor("oct", request)
    batch = _input_batch("oct")
    probability, p_amd = predictor._session.run(
        ["probability", "p_amd"], {ONNX_INPUT_NAME: batch}
    )
    np.testing.assert_allclose(p_amd, probability[:, 1], atol=1e-6)


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_feature_tap_shape_and_cross_session_determinism(
    module: str, request: pytest.FixtureRequest, weights_dir: Path
) -> None:
    predictor = _predictor(module, request)
    batch = _input_batch(module)
    feature_name = predictor._session.get_outputs()[-1].name
    (features_first,) = predictor._session.run([feature_name], {ONNX_INPUT_NAME: batch})
    assert features_first.shape == (2, FEATURE_DIM)
    fresh_session = _load_session(module, artifact_paths(module, weights_dir)["onnx"])
    (features_second,) = fresh_session.run([feature_name], {ONNX_INPUT_NAME: batch})
    np.testing.assert_array_equal(features_first, features_second)


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_predict_is_deterministic_across_runs(
    module: str, request: pytest.FixtureRequest
) -> None:
    predictor = _predictor(module, request)
    image = make_synthetic_oct(21) if module == "oct" else make_synthetic_fundus(21)
    first = predictor.predict(image)
    second = predictor.predict(image)
    assert first.probability == second.probability
    assert first.ood_score == second.ood_score
    assert first == second


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_predict_batch_matches_single_predictions(
    module: str, request: pytest.FixtureRequest
) -> None:
    predictor = _predictor(module, request)
    maker = make_synthetic_oct if module == "oct" else make_synthetic_fundus
    images = [maker(seed) for seed in (31, 32)]
    batched = predictor.predict_batch(images)
    singles = [predictor.predict(image) for image in images]
    for batch_result, single_result in zip(batched, singles):
        assert abs(batch_result.probability - single_result.probability) < 1e-6
        assert batch_result.decision == single_result.decision


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_prediction_result_contract(module: str, request: pytest.FixtureRequest) -> None:
    predictor = _predictor(module, request)
    image = make_synthetic_oct(41) if module == "oct" else make_synthetic_fundus(41)
    prediction = predictor.predict(image)
    assert prediction.module == module
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.decision == (prediction.probability >= prediction.threshold)
    assert prediction.ood_score > 0.0
    if module == "oct":
        assert prediction.class_probabilities is not None
        assert set(prediction.class_probabilities) == {"Normal", "AMD", "DME"}
        assert sum(prediction.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    else:
        assert prediction.class_probabilities is None
    if module == "dr":
        assert prediction.venn_abers_interval is None
    else:
        assert prediction.venn_abers_interval is not None


@pytest.mark.parametrize("module", ["dr", "glaucoma", "oct"])
def test_uniform_noise_is_flagged_out_of_distribution(
    module: str, request: pytest.FixtureRequest
) -> None:
    predictor = _predictor(module, request)
    rng = np.random.default_rng(99)
    noise = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    prediction = predictor.predict(noise)
    assert prediction.ood_flagged


def test_cli_predict_end_to_end_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    weights_dir: Path,
) -> None:
    image_path = tmp_path / "b_scan.jpeg"
    cv2.imwrite(
        str(image_path), cv2.cvtColor(make_synthetic_oct(51), cv2.COLOR_RGB2BGR)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "retguard",
            "predict",
            "--module",
            "oct",
            str(image_path),
            "--weights-dir",
            str(weights_dir),
            "--json",
        ],
    )
    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["module"] == "oct"
    assert 0.0 <= payload[0]["probability"] <= 1.0
