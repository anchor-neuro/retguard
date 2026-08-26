# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Deployment constants for the RETGUARD inference package.

Every value carries a citation to the paper section or deployment artifact
that fixes it. These values are pre-specified; do not tune them at inference.
"""

from typing import Final

# EfficientNet-V2-M input resolution used by all three modules (paper section 2.3;
# export_metadata.json "input_size": 480 for every export).
INPUT_SIZE: Final = 480

# ImageNet channel statistics (Deng et al. 2009), applied as
# (pixel - mean*255) / (std*255) on RGB input, matching the albumentations
# Normalize transform in the released evaluation code (train_binary.py MEAN/STD).
IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)

# Canonical module identifiers, used verbatim in the CLI, the weights registry,
# release asset names, and the model-card filenames.
MODULES: Final = ("dr", "glaucoma", "oct")

# DR module: temperature and referral threshold (paper section 2.4; both are
# recorded in the DR module's external_eval_binary.json evaluation artifact.
# The DR operating_point.json holds superseded thresholds 0.5084/0.2941 and is
# not the source of the deployed value).
# The DR module uses temperature scaling only - it has no Venn-Abers layer.
DR_TEMPERATURE: Final = 0.841517
DR_THRESHOLD: Final = 0.204983

# Glaucoma module: threshold from the glaucoma module's operating_point.json
# ("threshold"; the source tree spells the directory "Glucoma").
# Venn-Abers is the deployed calibrator. The fitted temperature is retained as
# an evaluation artifact, but packaged inference requires a valid Venn-Abers
# artifact and fails closed if it is missing or invalid.
GLAUCOMA_TEMPERATURE: Final = 0.94653
GLAUCOMA_THRESHOLD: Final = 0.044776

# OCT module: full-precision temperature from the OCT module's
# operating_point.json (source tree "AMD & DME"; the paper rounds it to 0.819). Venn-Abers on P(AMD) is the deployed
# calibrator; the temperature was fitted but not used for any reported
# operating point (paper section 2.4). The threshold is the pre-specified
# engineering ceiling max_clinical_threshold, which bound in the executed run
# (OCT model card section 8).
OCT_TEMPERATURE: Final = 0.819011
OCT_AMD_THRESHOLD: Final = 0.70

# Decimal places for rendering each module's decision threshold digit-for-digit
# as the paper and model cards print it (0.204983 / 0.044776 / 0.70; paper
# section 2.4). Python floats cannot carry the OCT trailing zero on their own.
THRESHOLD_DISPLAY_DECIMALS: Final = {"dr": 6, "glaucoma": 6, "oct": 2}

# OCT class order as exported: logit/probability columns are
# [Normal, AMD, DME]; the graph's p_amd output is probability[:, 1]
# (AMD & DME export_metadata.json "outputs").
OCT_CLASS_NAMES: Final = ("Normal", "AMD", "DME")
OCT_AMD_CLASS_INDEX: Final = 1

# OOD gate: Mahalanobis distance on L2-normalized 1,280-d post-GAP features,
# pooled Gaussian, empirical covariance + 1e-5 ridge, Cholesky solve, sqrt
# distance (ood_gate.py OODGate; paper section 2.4). Flag threshold is the
# 97th percentile of the shipped training_scores; the literals below are those
# percentiles as reported in paper Table (deployment constants) and are
# reproduced exactly (to 4 decimals) by the shipped ood_gate npz artifacts.
OOD_PERCENTILE: Final = 97.0
DR_OOD_THRESHOLD: Final = 47.3773
GLAUCOMA_OOD_THRESHOLD: Final = 48.3972
OCT_OOD_THRESHOLD: Final = 31.2676
OOD_RIDGE: Final = 1e-5
FEATURE_DIM: Final = 1280
# Floor applied when L2-normalizing feature rows (ood_gate.py _normalize).
FEATURE_NORM_FLOOR: Final = 1e-8

# Venn-Abers scalar rule p = p1 / (1 - p0 + p1): denominator floor
# (venn_abers.py predict; paper section 2.4 "floored at 1e-15").
VENN_ABERS_DENOMINATOR_FLOOR: Final = 1e-15
# Degenerate-slope guard in the CSD convex-hull slope computation
# (venn_abers.py _slope: a vertical segment yields an infinite slope).
VENN_ABERS_SLOPE_EPSILON: Final = 1e-30

# TTA view counts: fundus modules use the full D4 dihedral group; OCT uses
# identity + horizontal flip only, because rotations and vertical flips invert
# retinal layer ordering (tta_utils.py of each module; paper section 2.3).
FUNDUS_TTA_VIEWS: Final = 8
OCT_TTA_VIEWS: Final = 2

# Fundus preprocessing (preprocess.py preprocess_fundus, identical in the DR
# and glaucoma sources; "LOCKED" in both): circle crop -> resize -> Ben Graham
# unsharp -> CLAHE on LAB L -> circular mask.
FUNDUS_CROP_INTENSITY_THRESHOLD: Final = 10
FUNDUS_CROP_MIN_BOX_FRACTION: Final = 0.4
BEN_GRAHAM_GAIN: Final = 4
BEN_GRAHAM_OFFSET: Final = 128
BEN_GRAHAM_SIGMA_DIVISOR: Final = 30
CLAHE_CLIP_LIMIT: Final = 2.0
CLAHE_GRID: Final = (8, 8)
CIRCULAR_MASK_RADIUS_FRACTION: Final = 0.45

# OCT preprocessing (preprocess_oct.py): border crop -> resize -> grayscale ->
# percentile clip [1, 99] -> CLAHE -> 3-channel replication.
OCT_BORDER_CROP_THRESHOLD: Final = 10
OCT_BORDER_CROP_MIN_ROI_FRACTION: Final = 0.3
OCT_PERCENTILE_LOW: Final = 1.0
OCT_PERCENTILE_HIGH: Final = 99.0
# Below this clipped-range width the percentile step is skipped as degenerate
# (preprocess_oct.py _percentile_normalize).
OCT_PERCENTILE_MIN_RANGE: Final = 1.0

# ONNX I/O contract as smoke-tested (export_metadata.json of each module).
ONNX_INPUT_NAME: Final = "input"
MODULE_OUTPUT_NAMES: Final = {
    "dr": ("probability", "logit"),
    "glaucoma": ("probability", "logit"),
    "oct": ("probability", "logit", "p_amd"),
}

# Internal graph tensor holding the 1,280-d post-GAP features that feed the
# classifier head, exposed at session load by graph augmentation (REPO_BLUEPRINT
# section 3.3). Locked by one-time inspection of the shipped ONNX graphs.
GAP_OUTPUT_NODE: Final = {
    "dr": "/model/backbone/global_pool/flatten/Flatten_output_0",
    "glaucoma": "/model/backbone/global_pool/flatten/Flatten_output_0",
    "oct": "view",
}

# Internal tensor holding the last pre-GAP spatial feature map (N, 1280, 15, 15),
# used for exact analytic Grad-CAM; GAP of this tensor equals the pooled features
# (verified numerically against the shipped graphs, research_architecture.md §4).
CAM_SPATIAL_NODE: Final = {
    "dr": "/model/backbone/bn2/act/Mul_output_0",
    "glaucoma": "/model/backbone/bn2/act/Mul_output_0",
    "oct": "silu_146",
}
# Heatmap overlay recipe for the UI (ianpan blend convention, research_demos.md §3.9).
CAM_OVERLAY_ALPHA: Final = 0.4
# Names of the head initializers in the shipped graphs: GAP -> Gemm(1280->256)
# -> ReLU -> Gemm(256->K) (research_architecture.md §4.1-4.2). One-time
# inspection of all three shipped graphs confirmed the OCT export uses the
# same names as the fundus exports.
CAM_HEAD_INITIALIZERS: Final = (
    "model.binary_head.0.weight",
    "model.binary_head.0.bias",
    "model.binary_head.3.weight",
    "model.binary_head.3.bias",
)
