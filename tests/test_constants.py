# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Deployment constants match the paper's pre-specified values exactly."""

from retguard import constants


def test_module_registry_is_closed() -> None:
    assert constants.MODULES == ("dr", "glaucoma", "oct")


def test_input_contract() -> None:
    assert constants.INPUT_SIZE == 480
    assert constants.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert constants.IMAGENET_STD == (0.229, 0.224, 0.225)
    assert constants.ONNX_INPUT_NAME == "input"


def test_temperatures_and_thresholds_equal_paper_values() -> None:
    assert constants.DR_TEMPERATURE == 0.841517
    assert constants.DR_THRESHOLD == 0.204983
    assert constants.GLAUCOMA_TEMPERATURE == 0.94653
    assert constants.GLAUCOMA_THRESHOLD == 0.044776
    assert constants.OCT_TEMPERATURE == 0.819011
    assert constants.OCT_AMD_THRESHOLD == 0.70


def test_ood_constants_equal_paper_values() -> None:
    assert constants.DR_OOD_THRESHOLD == 47.3773
    assert constants.GLAUCOMA_OOD_THRESHOLD == 48.3972
    assert constants.OCT_OOD_THRESHOLD == 31.2676
    assert constants.OOD_PERCENTILE == 97.0
    assert constants.OOD_RIDGE == 1e-5
    assert constants.FEATURE_DIM == 1280


def test_tta_view_counts() -> None:
    assert constants.FUNDUS_TTA_VIEWS == 8
    assert constants.OCT_TTA_VIEWS == 2


def test_per_module_tables_cover_every_module_exactly() -> None:
    assert tuple(constants.MODULE_OUTPUT_NAMES) == constants.MODULES
    assert tuple(constants.GAP_OUTPUT_NODE) == constants.MODULES
    assert constants.MODULE_OUTPUT_NAMES["dr"] == ("probability", "logit")
    assert constants.MODULE_OUTPUT_NAMES["glaucoma"] == ("probability", "logit")
    assert constants.MODULE_OUTPUT_NAMES["oct"] == ("probability", "logit", "p_amd")


def test_oct_class_order() -> None:
    assert constants.OCT_CLASS_NAMES == ("Normal", "AMD", "DME")
    assert constants.OCT_CLASS_NAMES[constants.OCT_AMD_CLASS_INDEX] == "AMD"
