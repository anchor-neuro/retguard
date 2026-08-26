# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning:
retrained weights are a minor bump with fresh checksums; documentation or
metric corrections without weight changes are a patch.

## [Unreleased]

## [1.1.1] - 2026-08-27

No model weights changed. The v1.0.0 model archives and their SHA-256 digests
remain the only release assets.

### Corrected

- Aligned documentation with the released runtime: glaucoma and OCT require
  their Venn-Abers artifacts and fail closed when calibration artifacts are
  missing or invalid; there is no temperature-scaling fallback in packaged
  inference.
- Clarified that the OCT Boolean output is an AMD-only decision at
  `P(AMD) >= 0.70`, not a DME-inclusive referral decision.
- Removed public Hugging Face mirror/Space claims pending anonymous endpoint
  verification, and changed the preprint status to manuscript in preparation.
- Corrected model-card audit wording, the DR ONNX parity disposition, the ORIGA
  literature comparison, and dataset-rights summaries.
- Withdrew all archived Duke/Srinivasan OCT performance, calibration, OOD, DME,
  threshold, confusion-matrix, and volume-level claims after the released model
  failed the prespecified reproduction gate on all 3,231 scans. No replacement
  performance estimate was generated.

## [1.1.0] - 2026-08-23

Strictly additive. The v1.0.0 release assets, their SHA-256 digests, the
download path, and the `PredictionResult` contract are unchanged; nothing is
listed under Changed or Removed because nothing changed in weights or
evaluation.

### Added

- `retguard serve`: a local research-demo browser UI (Gradio Blocks; binds
  127.0.0.1, analytics disabled, uploads purged within about a minute),
  behind the new `[ui]` optional dependency set.
- Exact analytic Grad-CAM computed from the shipped ONNX graphs, exposed as
  `Predictor.predict_with_cam`. Verified against the torch reference
  pipeline: Spearman rank correlation 1.000000 on all 15 image/module pairs
  (gate: at least 0.95 per image).
- Single-view option `predict(..., tta=False)`, documented as a deviation
  from the published test-time-augmentation protocol.

## [1.0.0] - 2026-08-23

### Added

- `retguard` inference package: per-module preprocessing (Ben Graham fundus
  pipeline; OCT border-crop/percentile/CLAHE pipeline), logit-space TTA,
  temperature and Venn-Abers calibration, Mahalanobis OOD gate, and the three
  pre-specified decision thresholds, with a `retguard` CLI
  (`download` / `verify` / `predict`).
- Three ONNX model releases (dr, glaucoma, oct) distributed as zips via GitHub
  Releases, with SHA-256 verification of every zip and member file.
- Model cards for all three modules, documenting evaluation, calibration, OOD
  behavior, and every known limitation and failure mode.
- Manuscript figures, with SVG sources for Figures 1, 3, and 4 and a PNG source
  for Figure 2.
- Research-use license structure: PolyForm Noncommercial 1.0.0 (code),
  CC BY-NC 4.0 (weights, companion artifacts, figures, model cards,
  documentation), and a commercial-licensing rider with a 90-day evaluation
  permission.

[1.1.1]: https://github.com/anchor-neuro/retguard/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/anchor-neuro/retguard/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0
