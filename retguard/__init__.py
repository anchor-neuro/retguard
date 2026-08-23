# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""RETGUARD: calibrated, OOD-aware retinal screening inference.

See the repository README for installation, quickstart, and license terms.
"""

from retguard.predictor import PredictionResult, Predictor, load

__version__ = "1.1.0"
__all__ = ["PredictionResult", "Predictor", "__version__", "load"]
