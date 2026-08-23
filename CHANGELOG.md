# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning:
retrained weights are a minor bump with fresh checksums; documentation or
metric corrections without weight changes are a patch.

## [1.0.0] - 2026-08-23

### Added

- `retguard` inference package: per-module preprocessing (Ben Graham fundus
  pipeline; OCT border-crop/percentile/CLAHE pipeline), logit-space TTA,
  temperature and Venn-Abers calibration, Mahalanobis OOD gate, and the three
  pre-specified referral thresholds, with a `retguard` CLI
  (`download` / `verify` / `predict`).
- Three ONNX model releases (dr, glaucoma, oct) distributed as zips via GitHub
  Releases, with SHA-256 verification of every zip and member file.
- Model cards for all three modules, documenting evaluation, calibration, OOD
  behavior, and every known limitation and failure mode.
- Manuscript figures with SVG sources.
- Research-use license structure: PolyForm Noncommercial 1.0.0 (code),
  CC BY-NC 4.0 (weights, companion artifacts, figures, model cards,
  documentation), and a commercial-licensing rider with a 90-day evaluation
  permission.

[1.0.0]: https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0
