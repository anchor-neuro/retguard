# RETGUARD

Calibrated, out-of-distribution-aware retinal screening models — referable diabetic retinopathy and glaucoma from fundus photographs, AMD/DME from OCT B-scans — released as ONNX exports (per-module export-verification status is disclosed in the model cards) with the full candidate deployment-safety stack: temperature/Venn-Abers calibration, a Mahalanobis OOD gate, and pre-specified referral thresholds.

[![License](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0%20%2B%20CC%20BY--NC%204.0-lightgrey)](LICENSE.md)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)
[![CI](https://github.com/anchor-neuro/retguard/actions/workflows/ci.yml/badge.svg)](https://github.com/anchor-neuro/retguard/actions/workflows/ci.yml)

Preprint (medRxiv, in submission — see [Citation](#citation)) · [Model cards](docs/) · [Weights (Releases)](https://github.com/anchor-neuro/retguard/releases) · [Commercial licensing](COMMERCIAL-LICENSE.md)

![RETGUARD system overview: three per-disease classifiers sharing one deployment-safety stack](figures/F1_system_overview.png)

## Results

| Module | Modality | Task | Primary external result |
|---|---|---|---|
| dr | Fundus | Referable diabetic retinopathy | Messidor-2 (n=1,744, zero-shot): AUC 0.9691 (95% CI 0.9595–0.9772); sensitivity 97.16% (95.45–98.51%), specificity 76.07% (73.71–78.32%) at the pre-specified threshold |
| glaucoma | Fundus | Referable glaucoma | REFUGE-Val (n=400, zero-shot): AUC 0.9278 (0.8642–0.9763); ORIGA (n=650, zero-shot): AUC 0.8729 (0.8399–0.9028), both under minimal preprocessing |
| oct | OCT B-scan | Normal / AMD / DME | Duke Srinivasan (n=3,231, zero-shot): AMD-vs-rest AUC 0.9883 (0.9845–0.9915); OCTID (n=261, zero-shot; 55 AMD positives, no DME cases): AUC 0.9999 (0.9995–1.0000) |

CIs are bootstrap percentile intervals. Full evaluation — internal, dataset-family-holdout, and failure-mode results — is in the paper and the [model cards](docs/).

## Known failure modes

Reported with the same prominence as the results above, because they define where the system must not be used:

- The glaucoma module fails on optic-disc crops: on RIM-ONE DL it performs below chance (AUC 0.3667). It is a full-fundus classifier; cropped inputs are out of intended use.
- Cropped fields of view degrade it: ACRIMA AUC 0.7185 (minimal preprocessing) / 0.6049 (training-domain preprocessing).
- The OCT module's "Normal" output means "no AMD or DME detected", not "healthy": in a 354-image confounder test (epiretinal membrane, retinal artery/vein occlusion, vitreomacular interface disease), 347 of 354 non-target pathologies were output as Normal.
- Zero-shot external DME evidence rests on one dataset (Duke), where B-scan-level DME sensitivity was 77.38%.
- The DR module's calibration exceeds the project's own 0.05 expected-calibration-error gate (calibration-set ECE 0.052537; 0.0811 on Messidor-2).
- One glaucoma result (ORIGA under training-domain preprocessing, AUC 0.9988) exceeds every published figure and is reported as unconfirmed pending independent replication.
- The DR module's ONNX export failed its logit-parity tolerance (maximum logit difference 0.002526 against a 0.001 threshold); probability parity passed at 0.000626, so the practical impact on threshold decisions is bounded, and the failure is disclosed in the DR model card.

## Installation

```
pip install git+https://github.com/anchor-neuro/retguard
retguard download --module all
```

Supported environment: Python 3.10–3.12, CPU-only inference via onnxruntime. No GPU is required.

## Quickstart

No sample images are bundled; supply your own retinal OCT B-scan. The Kermany OCT dataset (Mendeley Data, doi:10.17632/rscbjbr9sj.2, CC BY 4.0) is a public source. With a B-scan saved as `oct_bscan.jpeg`:

```python
import retguard

predictor = retguard.load("oct")
result = predictor.predict("oct_bscan.jpeg")
print(result.class_probabilities, result.decision, result.ood_flagged)
```

The CLI equivalent:

```
retguard download --module oct
retguard predict --module oct oct_bscan.jpeg --json
```

Runnable scripts for both modalities are in [`examples/`](examples/).

## Model weights

| Module | Release asset | Size (bytes) | SHA-256 of the model file |
|---|---|---|---|
| dr | `retguard-dr-v1.0.0.zip` | 204,707,438 | `retguard_dr_v1.0.0.onnx`: `f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b` |
| glaucoma | `retguard-glaucoma-v1.0.0.zip` | 204,819,003 | `retguard_glaucoma_v1.0.0.onnx`: `b5686229ecc9050caddf61a66686e77e3e5cb4085425cca881078acb586cbb7b` |
| oct | `retguard-oct-v1.0.0.zip` | 204,678,963 | `retguard_oct_v1.0.0.onnx`: `5ebd2e814a718edca922c26f5bdce380c6b38506f923103a8cb3362b67fb75f3`; `retguard_oct_v1.0.0.onnx.data`: `88d6c4d6803d3201b182eeb528b9ab08f2641de2b1d1ef672c807f9ecda5243c` |

Each zip additionally contains the module's OOD-gate `.npz`, the Venn-Abers calibrator `.npz` where the module deploys one (glaucoma, oct), `LICENSE.txt`, and the module's model card. Weights are distributed via the [v1.0.0 release](https://github.com/anchor-neuro/retguard/releases/tag/v1.0.0), never via the git tree. SHA-256 digests for every zip and member file are published in the release's `SHA256SUMS.txt` and embedded in `retguard/weights.py`; `retguard download` verifies them automatically, and `retguard verify` re-checks an existing installation.

## Model cards

- [`docs/MODEL_CARD_dr.md`](docs/MODEL_CARD_dr.md) — referable diabetic retinopathy from fundus photographs.
- [`docs/MODEL_CARD_glaucoma.md`](docs/MODEL_CARD_glaucoma.md) — referable glaucoma from full fundus photographs.
- [`docs/MODEL_CARD_oct.md`](docs/MODEL_CARD_oct.md) — three-class Normal/AMD/DME from OCT B-scans.

The cards document every known limitation and every value the independent audit could not trace to an artifact.

## Intended use and limitations

RETGUARD is a research artifact. It is not a medical device, has no regulatory clearance in any jurisdiction, and must not be used for clinical diagnosis, screening, triage, or patient management. All published evidence is retrospective and computed on public research datasets; no prospective validation and no validation on a target camera or OCT device has been performed. The intended-use sections of the model cards govern.

## License

| Artifact class | License | File |
|---|---|---|
| Source code | PolyForm Noncommercial 1.0.0 | [LICENSE.md](LICENSE.md) |
| Weights, companion artifacts, figures, model cards, documentation | CC BY-NC 4.0 | [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md) |
| Commercial use | Separate written agreement | [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) |

This repository is source-available for research use, released under a research-use (noncommercial) license structure. A 90-day internal-evaluation permission for organizations considering a commercial license is included in [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

The released weights were trained on public research datasets whose terms permit research use; several do not permit commercial redistribution. Any commercial deployment requires retraining on appropriately licensed data. Per-dataset provenance and license status are given in the paper's Methods and Data Availability sections.

## Citation

```bibtex
@article{aboelmaaty2026retguard,
  author  = {Aboelmaaty, Sameh},
  title   = {{RETGUARD}: Calibrated, Out-of-Distribution-Aware Deep Learning for
             Multi-Disease Retinal Screening from Fundus Photography and Optical
             Coherence Tomography, with External Validation and Explicit
             Failure-Mode Reporting},
  journal = {medRxiv},
  year    = {2026}
}
```

Or use GitHub's "Cite this repository" button, which reads [CITATION.cff](CITATION.cff).

## Acknowledgements and contact

This work builds on public datasets: EyePACS, DDR, APTOS, IDRiD, DeepDRiD, Messidor-2, AIROGS, G1020, REFUGE, ORIGA, DRISHTI-GS, FIVES, RIM-ONE DL, ACRIMA, Kermany, Noor, OCTDL, Duke/Srinivasan, and OCTID. Each dataset's terms are credited in the paper.

Sameh Aboelmaaty — ghoneim2012@gmail.com — Anchor Neuro, Delaware, USA.

Competing interests: the author developed RETGUARD and founded Anchor Neuro, which may commercialize the system.
