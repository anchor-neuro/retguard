# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Classify OCT B-scans with the oct module.

Usage: python predict_oct.py IMAGE [IMAGE ...]

Suitable retinal OCT B-scans are available in the Kermany OCT dataset
(Mendeley Data, doi:10.17632/rscbjbr9sj.2, CC BY 4.0). Run
'retguard download --module oct' once before this script.
"""

import sys

import retguard

predictor = retguard.load("oct")
for image_path in sys.argv[1:]:
    prediction = predictor.predict(image_path)
    assert prediction.class_probabilities is not None
    classes = ", ".join(
        f"{name}={value:.4f}" for name, value in prediction.class_probabilities.items()
    )
    print(
        f"{image_path}  {classes}  p_amd={prediction.probability:.4f}  "
        f"decision={prediction.decision}  ood_flagged={prediction.ood_flagged}"
    )
