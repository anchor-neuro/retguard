# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Run the DR and glaucoma modules on one fundus photograph.

Usage: python predict_fundus.py IMAGE

No fundus image is bundled: the fundus training and evaluation datasets do
not permit redistribution. Public sources include Messidor-2 (via ADCIS) and
APTOS 2019 (via Kaggle), each under its own license acceptance. Run
'retguard download --module dr' and '--module glaucoma' once before this script.
"""

import sys

import retguard

image_path = sys.argv[1]
for module in ("dr", "glaucoma"):
    prediction = retguard.load(module).predict(image_path)
    print(
        f"{module}: probability={prediction.probability:.6f}  "
        f"decision={prediction.decision}  ood_score={prediction.ood_score:.2f}  "
        f"ood_flagged={prediction.ood_flagged}"
    )
