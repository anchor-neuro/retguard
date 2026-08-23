# Model Card — RETGUARD Diabetic Retinopathy Module (Fundus)

**Model card version:** 1.0 | **Date:** 2026-08-22 | **Status:** Research software. No regulatory clearance.

> **This is research software. It is not a medical device. It has no FDA clearance, no CE mark, and no regulatory authorization in any jurisdiction. It must not be used to make or inform any clinical decision about any patient.**

This card follows the Model Cards for Model Reporting framework (Mitchell M, Wu S, Zaldivar A, et al. Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*. 2019:220-229. DOI: 10.1145/3287560.3287596).

Every number in this card is copied verbatim from the RETGUARD manuscript (v1.2, 2026-08-22) or from the independent verification manifest `manifest_DR.json` (generated 2026-08-22). Nothing is estimated, and nothing is rounded beyond the precision the source artifact stores. Values the audit could not trace to an artifact are labeled **report-stated** wherever they appear.

---

## 1. Model details

| Item | Value |
|---|---|
| Developer | Sameh Aboelmaaty, Anchor Neuro, Delaware, USA |
| Model card date | 2026-08-22 |
| Module version | RETGUARD DR module, v6 (training pipeline run `pipeline_20260306_155744`, 2026-03-06) |
| Model type | Binary image classifier (referable diabetic retinopathy vs. non-referable) |
| Architecture | EfficientNetV2-M (`tf_efficientnetv2_m.in21k_ft_in1k`, timm), pretrained on ImageNet-21k and fine-tuned on ImageNet-1k |
| Input | Single color fundus photograph, 480 x 480 x 3 |
| Trainable parameters | 53,186,549 (verified from pipeline log) |
| Output | Calibrated probability of referable DR; binary decision at the pre-specified threshold 0.204983 |
| Calibration | Temperature scaling only (temperature 0.841517). **No Venn-Abers layer in this module**, unlike the glaucoma and OCT modules |
| Test-time augmentation | 8 views, the full D4 dihedral group (verified from released source; paper Table 2) |
| Out-of-distribution gate | Mahalanobis-distance detector on penultimate features, 97th-percentile threshold 47.3773; advisory only |
| Export format | ONNX, opset 17, dynamic batch, input shape (batch, 3, 480, 480); file size 212,474,067 bytes measured on disk |
| Export status | **Numeric verification FAILED the logit-parity tolerance** (see Section 10) |
| Training seed | 42 (single run) |
| License — weights | CC BY-NC 4.0, research use; commercial licensing available separately (see LICENSE-WEIGHTS.md, COMMERCIAL-LICENSE.md) |
| License — code | PolyForm Noncommercial 1.0.0 (see LICENSE.md) |
| Contact | ghoneim2012@gmail.com |
| Repository | https://github.com/anchor-neuro/retguard |

**Paper citation.** Aboelmaaty S. RETGUARD: Calibrated, Out-of-Distribution-Aware Deep Learning for Multi-Disease Retinal Screening from Fundus Photography and Optical Coherence Tomography, with External Validation and Explicit Failure-Mode Reporting. Manuscript v1.2, 2026-08-22. Anchor Neuro, Delaware, USA. Submission-ready draft (medRxiv).

**Verified training configuration** (from pipeline log and `history.csv`): loss — focal (gamma 2.0) on approximately 70% of batches, binary cross-entropy on approximately 30% MixUp batches; label smoothing epsilon 0.1 (targets 0 to 0.05, 1 to 0.95); precision BF16; drop-path 0.2; EMA decay 0.9998; maximum epochs 40, epochs run 25, best epoch 18 (best validation AUC 0.9444); early stopping patience 7, AUC-based, triggered at epoch 25; checkpoint policy — highest AUC with F1 >= 0.73.

**Configuration items not recorded in any evaluation artifact, subsequently recovered from the released source code by an independent audit (paper Table 2, cells marked *(src)*):** AdamW (backbone learning rate 1e-4, head 3e-4; linear warmup then cosine decay), MixUp alpha 0.3 applied with probability 0.3, dropout 0.3, batch size 16 with 4-step gradient accumulation, classification head 1,280-256-1, and the 8-view D4 TTA composition.

---

## 2. Intended use

**Primary intended use.** Research and methodological evaluation only: retrospective benchmarking of referable-DR classification, calibration behavior, out-of-distribution flagging, and operating-point portability on public fundus datasets.

**Primary intended users.** Machine-learning researchers and clinical-AI evaluators working with de-identified, publicly available retinal image datasets.

**Required input type.** A single color fundus photograph of the posterior pole, resized to 480 x 480. The module is a full-fundus-image classifier.

**Out of scope — this module must not be used for:**

- Any clinical decision, triage, referral, diagnosis, or exclusion of disease for any patient.
- Any use implying regulatory clearance. There is none, in any jurisdiction.
- Any imaging type other than a color fundus photograph — not optic-disc crops, not aggressively cropped fields of view, not OCT, not any other modality.
- Excluding diabetic macular edema. The source labels encode DR severity only; a non-referable output does **not** exclude DME (Section 11, item 2; Section 12, item 8).
- Any autonomous or unattended operation. There is no gradability/imageability pathway (Section 12, item 17).
- Deployment on any device or in any population without prospective, target-hardware validation and local recalibration.

---

## 3. Factors

**Populations and acquisition settings represented in training** (per Table 1 of the manuscript and the pipeline log):

| Dataset | Origin / device | Role |
|---|---|---|
| EyePACS (Kaggle DR 2015) | US telemedicine network; mixed cameras | Train + internal evaluation |
| DDR | China, 147 hospitals, 42 camera types | Train only |
| APTOS 2019 | India (Aravind Eye Hospital) | Train + evaluation (dataset-family holdout) |
| IDRiD | India | Train + evaluation (dataset-family holdout) |
| DeepDRiD | China, multi-site | Train + evaluation (dataset-family holdout) |

**Population represented in strictly external evaluation:** Messidor-2 — France, primary care, 2005-2010.

**Known geographic and ethnic gaps.** Training and evaluation data derive predominantly from US, Chinese, Indian, and French cohorts. **African populations are essentially absent from both training and evaluation.** Performance in those populations is unknown, not merely unproven. No per-subgroup analysis by age, sex, ethnicity, or camera model was performed, because the public datasets used do not carry the necessary metadata in the artifacts audited.

**Device factors.** Camera model, field definition, image resolution, mydriasis status, and media clarity are not recorded per image in the audited artifacts and were not analyzed as factors. The OOD gate measures feature-space distance only; it is not a camera-compatibility test and not a gradability test.

**Instrumentation factor, resolved from source.** The deployed input-preprocessing recipe for this module is not evidenced by artifacts of the audited run (the preprocessing phase was skipped there; Section 12, item 5), but it has since been fully reconstructed from the released source code: it is byte-for-byte the same five-step LOCKED fundus function the glaucoma module records in its export metadata (paper Section 2.2, verified from released source), and the accompanying inference package ships that recipe parity-verified.

---

## 4. Metrics

| Metric | Why it is reported |
|---|---|
| AUC with 95% bootstrap CI | Threshold-free discrimination; the only summary unaffected by threshold selection, and therefore the only portable comparator across datasets |
| Sensitivity and specificity at the pre-specified threshold (0.204983), with 95% bootstrap CIs | The deployment-relevant operating characteristic; reported at a threshold fixed before any benchmark was seen |
| Sensitivity at 90% and 95% specificity | Threshold-free operating-point comparison against published systems that report at fixed specificity |
| PPV, NPV, F1, and the full confusion matrix | Referral-workload interpretation; PPV and NPV are prevalence-dependent and are reported with the cohort prevalence stated |
| Expected calibration error (ECE), maximum calibration error (MCE), Brier score | Calibration is a safety property: miscalibrated probabilities corrupt any fixed referral threshold. Reported separately for the calibrator's own fitting partition (in-sample, optimistically biased) and for held-out partitions |
| OOD flag rate per dataset | Measures how often the advisory gate would route an input to human review. Reported as a result, never tuned |
| Sight-threatening sensitivity | Sensitivity within the severe subset of referable disease. **Caveat: the grade boundary defining this subset is not recorded in the audited artifacts** |

**Statistical protocol.** All CIs are bootstrap-based, of the percentile type, unstratified, with fixed seeds (verified from released source; paper Section 2.7). The operating-point threshold, sensitivity, and specificity CIs use 2,000 resamples (seed 42); every per-dataset AUC, sensitivity, and specificity CI uses 1,000 resamples (seeds 42, 43, and 44 respectively) — the 1,000-resample count falls short of the project's own 2,000-resample reporting convention and is disclosed as such (Section 12, item 12). ECE was computed with 15 bins where the bin count is recorded (the calibration block).

---

## 5. Training data

**Composition actually loaded by the training script** (pipeline log, lines 22-39):

| Dataset | n | Referable | Non-referable | Label treatment |
|---|---|---|---|---|
| EyePACS | 42,948 | 7,781 | 35,167 | cleanlab-cleaned |
| DDR | 8,173 | 3,177 | 4,996 | cleanlab-cleaned |
| APTOS | 1,844 | 514 | 1,330 | decontaminated train split |
| IDRiD | 320 | 190 | 130 | — |
| DeepDRiD | 543 | 258 | 285 | cleanlab-cleaned |
| **Total** | **53,828** | **11,920** | **41,908** | class ratio 41,908/11,920 = 3.516 |

**Messidor-2 was excluded from training** (pipeline log line 28). See Section 12, item 10, for a stale split file that contradicts this on its face.

**Model-selection partition:** train 53,828 / validation 5,499 (EyePACS model-select partition); patient-overlap check 0; laterality gap 0.0018.

**Label cleaning.** Confident learning (cleanlab) with an MLP probe on 1,280-dimensional backbone features and cross-validation was applied to EyePACS, DDR, and DeepDRiD before training; the APTOS train split was decontaminated. Per-dataset cleaning counts for the DR module are not separately stored in the audited artifacts. The label-quality rationale follows the Messidor-2 replication study, which attributed most of a 0.14-AUC gap primarily to label quality.

**Case definition.** Referable DR was defined by dichotomizing the source datasets' severity grades. **The exact grade cutoff used by the label-preparation pipeline is not recorded in the audited artifacts**, nor is the grade boundary defining "sight-threatening" disease, nor whether DME forms part of the referable composite. These must be documented from the label-preparation code.

**License and commercial-use status of the training and evaluation data:**

| Dataset | License / access terms |
|---|---|
| EyePACS (Kaggle DR 2015) | Kaggle competition terms |
| DDR | Research use (GitHub) |
| APTOS 2019 | Kaggle competition terms |
| IDRiD | CC BY 4.0 (per dataset paper) |
| DeepDRiD | Challenge terms |
| Messidor-2 | ADCIS research license |

None of these access terms was independently verified against the source repositories by the audit. **Commercial use of a model trained on these datasets is not established as permitted and must be reviewed against each dataset's terms before any such use.**

---

## 6. Evaluation data

Classification uses exactly the manuscript's terms. A dataset is **external (zero-shot)** only if no image or sibling split of that dataset contributed to training. A dataset whose train split contributed to training, but whose evaluated split is disjoint at the patient or image level, is a **dataset-family holdout** — this is *not* external validation.

| Evaluation set | n evaluated | Classification (manuscript's exact term) |
|---|---|---|
| EyePACS-Val | 7,857 (1,466 referable / 6,391 non-referable) | Internal held-out partition |
| EyePACS-Test | 47,379 (8,913 / 38,466) | Internal held-out partition |
| APTOS eval split | 461 (129 / 332) | Dataset-family holdout |
| IDRiD-Test | 71 (45 / 26) | Dataset-family holdout |
| DeepDRiD-Val | 302 (138 / 164) | Dataset-family holdout (0 patient overlap) |
| Messidor-2, 629-image subset | 629 (249 / 380) | External |
| **Messidor-2, full benchmark** | **1,744 evaluated of 1,748 (457 / 1,287); 4 excluded as ungradable** | **External (zero-shot)** |
| Calibration partition | 2,358 EyePACS images | In-sample for the fitted temperature |

**Split verification.** `verification_report.json` records 5/5 checks passed, 0 overlapping images, and 0 overlapping patients, covering APTOS, Messidor-2, IDRiD, and DeepDRiD only. **No split-verification artifact covers the EyePACS or DDR train/validation/test partitions.** The external-evaluation artifact itself carries leakage warnings stating that IDRiD-Test and DeepDRiD-Val "were used in training — metrics may be inflated"; the split-verification artifact explains these as dataset-family warnings rather than image-level leakage, but the rows remain dataset-family holdouts and not external validation.

**Provenance gap.** The provenance of the Messidor-2 referable-DR reference labels — which public grade set, whose grading protocol, whether DME was included, and who adjudicated the 4 ungradable exclusions — is not documented in the audited artifacts.

**629-image subset provenance.** The 629-image Messidor-2 partition is carried from a pre-existing split file that predates this training run; its selection provenance is not documented. It is retained here only because the sight-threatening breakdown is stored for it alone.

---

## 7. Quantitative results

All rows below are at the **pre-specified threshold 0.204983** with 8-view TTA, temperature 0.841517.

### 7.1 Internal, dataset-family-holdout, and external-subset evaluation

| Dataset | Exposure | n (ref/non-ref) | AUC (95% CI) | Sens. | Spec. | Sight-threatening sens. | Brier | ECE | OOD flagged |
|---|---|---|---|---|---|---|---|---|---|
| EyePACS-Val | internal holdout | 7,857 (1,466/6,391) | 0.9475 (0.9400-0.9543) | 0.9018 | 0.8341 | 0.9825 (336/342) | 0.062964 | 0.051146 | 5.26% |
| EyePACS-Test | internal holdout | 47,379 (8,913/38,466) | 0.9549 (0.9522-0.9574) | 0.9148 | 0.8465 | 0.9884 (2,040/2,064) | 0.056862 | 0.048999 | 4.82% |
| APTOS eval split | dataset-family holdout | 461 (129/332) | 0.9928 (0.9881-0.9969) | 1.0000 | 0.8735 | 1.0000 (38/38) | 0.047682 | 0.059102 | 4.99% |
| IDRiD-Test | dataset-family holdout | 71 (45/26) | 0.9427 (0.8779-0.9828) | 0.8667 | 0.7308 | 0.9167 (22/24) | 0.092654 | 0.118801 | 16.9% |
| DeepDRiD-Val | dataset-family holdout | 302 (138/164) | 0.9774 (0.9625-0.9901) | 0.9855 | 0.7561 | 0.9872 (77/78) | 0.079854 | 0.081973 | 10.26% |
| Messidor-2 (629 subset) | external | 629 (249/380) | 0.9673 (0.9530-0.9800) | 0.9759 | 0.7263 | 1.0000 (63/63) | 0.076363 | 0.065391 | 1.75% |

Small-sample warnings apply to the APTOS, IDRiD, and DeepDRiD rows; a wide-CI warning additionally applies to IDRiD-Test.

Additional stored values: EyePACS-Val PPV 0.5550, NPV 0.9737, F1 0.6871, sens@90spec 0.8649, sens@95spec 0.7974, confusion TP 1,322 / TN 5,331 / FP 1,060 / FN 144. EyePACS-Test PPV 0.5800, NPV 0.9772, F1 0.7099, sens@90spec 0.8856, sens@95spec 0.8290, confusion TP 8,154 / TN 32,561 / FP 5,905 / FN 759.

### 7.2 Primary external result — Messidor-2, full benchmark (n = 1,744, zero-shot)

| Quantity | Value |
|---|---|
| AUC (95% CI) | **0.9691 (0.9595-0.9772)** |
| Sensitivity at 0.204983 (95% CI) | **0.9716 (0.9545-0.9851)** |
| Specificity at 0.204983 (95% CI) | **0.7607 (0.7371-0.7832)** |
| Confusion matrix | TP 444, FN 13, FP 308, TN 979 |
| PPV / NPV / F1 | 0.5904 / 0.9869 / 0.7345 |
| Sensitivity at 90% specificity | 0.9103 |
| Sensitivity at 95% specificity | 0.8687 |
| Brier | 0.0702 |
| ECE | 0.0811 |

Referable prevalence in this benchmark is 26.2% (457/1,744), well above typical primary-care screening prevalence. PPV and NPV are prevalence-dependent: the PPV of 0.5904 would be substantially lower, and the NPV higher, in a real screening population. In referral-workload terms, this benchmark produced 444 true referrals against 308 false referrals at the pre-specified threshold. A prevalence-standardized projection of PPV/NPV has not been computed.

### 7.3 Post-hoc operating points — in-sample, subordinate to Section 7.2

**These three operating points were derived on the Messidor-2 benchmark itself and their sensitivity/specificity are then reported on that same benchmark. Threshold selection and performance measurement used the same data. This is circular. These values are not claims of portable operating performance and are subordinate to the pre-specified-threshold result in Section 7.2.** AUC is threshold-free and is unaffected.

| Point | Threshold | Sens. | Spec. | PPV | NPV | F1 | Youden J | Confusion (TP/FN/FP/TN) |
|---|---|---|---|---|---|---|---|---|
| OP-1 "Balanced" (in-sample) | 0.5084 | 0.9103 | 0.8990 | 0.7619 | 0.9658 | 0.8295 | 0.8093 | 416/41/130/1,157 |
| OP-2 "High-sensitivity" (in-sample) | 0.2941 | 0.9519 | 0.8159 | 0.6473 | 0.9795 | 0.7706 | 0.7678 | 435/22/237/1,050 |
| OP-3 "Moderate" (in-sample) | 0.3825 | 0.9278 | 0.8539 | 0.6928 | 0.9708 | 0.7933 | 0.7817 | 424/33/188/1,099 |
| OP-4 = pre-specified threshold | 0.205 | 0.9716 | 0.7607 | 0.5904 | 0.9869 | 0.7345 | 0.7323 | 444/13/308/979 |

The portable summaries are the AUC (0.9691) and the sensitivity-at-fixed-specificity values (0.9103 at 90% specificity; 0.8687 at 95% specificity).

### 7.4 Threshold selection (pre-specified, on the calibration partition)

Method: maximize specificity subject to sensitivity >= 0.90 (the target is not stored in the audited artifacts; the rule and its 0.90 constraint were recovered from the released source code — paper Section 2.4). Threshold 0.204983, selected on 2,358 EyePACS samples. On that selection set: sensitivity 0.9013, specificity 0.8279, AUC 0.949029 (95% CI 0.9378-0.9595). Bootstrap over 2,000 valid resamples: threshold mean 0.210155, SD 0.042703; sensitivity CI 0.9000-0.9069; specificity CI 0.7708-0.8839.

### 7.5 Release gate

The automated release gate passed on all six eligible evaluation datasets (APTOS, DeepDRiD-Val, EyePACS-Test, EyePACS-Val, IDRiD-Test, Messidor-2) with zero contaminated datasets recorded and zero failures. The pipeline itself flagged a warning: a 0.0501 spread between the best and worst evaluation-dataset AUC (APTOS 0.9928 to IDRiD-Test 0.9427). The numeric gate threshold values are not stored in the artifacts.

---

## 8. Calibration

| Item | Value |
|---|---|
| Calibrator type | Temperature scaling only. **No Venn-Abers layer** (unlike the glaucoma and OCT modules) |
| Temperature | 0.841517 |
| Fitting partition | 2,358 EyePACS images, 8-view TTA |
| ECE bins | 15 |

**In-sample assessment** (computed on the same partition used to fit the temperature; optimistically biased by construction): Brier 0.065162, **ECE 0.052537**, MCE 0.291673.

**Held-out assessment** (partitions the calibrator never saw):

| Partition | ECE | Brier |
|---|---|---|
| EyePACS-Val | 0.051146 | 0.062964 |
| EyePACS-Test | 0.048999 | 0.056862 |
| APTOS eval split | 0.059102 | 0.047682 |
| IDRiD-Test | 0.118801 | 0.092654 |
| DeepDRiD-Val | 0.081973 | 0.079854 |
| **Messidor-2, full (external)** | **0.0811** | 0.0702 |

**The in-sample calibration-set ECE of 0.052537 exceeds the project's own 0.05 deployment gate.** It does so on the partition where calibration is measured most favorably. Calibration then degrades further under distribution shift, reaching ECE 0.0811 on Messidor-2 and 0.118801 on IDRiD-Test. Deployed use would require local recalibration; none has been performed or validated.

---

## 9. Out-of-distribution gate

| Item | Value |
|---|---|
| Method | Mahalanobis distance on penultimate backbone features |
| Feature dimension | 1,280 (no evaluation artifact states it directly; the dimension is hard-asserted in the released OOD-gate source code — paper Section 2.4 — and supported indirectly by the cleanlab probe architecture) |
| Threshold percentile | 97th percentile of training-set scores |
| Threshold value | 47.3773 |
| Training-set score range (from log) | 16.01 to 85.42 |
| Behavior | **Advisory only.** Flags inputs for human review; does not alter or suppress predictions |

**Flag rates per evaluation set:**

| Dataset | Flag rate |
|---|---|
| Messidor-2 (629 subset) | 1.75% |
| EyePACS-Test | 4.82% |
| APTOS eval split | 4.99% |
| EyePACS-Val | 5.26% |
| DeepDRiD-Val | 10.26% |
| IDRiD-Test | 16.9% |

**What was and was not tested.** Flag rates were measured on the six evaluation sets above, and are reported as results, never tuned. The detector's exact form is not recorded in the evaluation artifacts but is now specified from the released source (paper Section 2.4): a single pooled Gaussian — not class-conditional — over L2-normalized 1,280-dimensional post-GAP features, with the plain empirical covariance plus a 1e-5 ridge, scored as the Mahalanobis distance (not its square) by Cholesky forward substitution. No flag rate was recorded for the full 1,744-image Messidor-2 benchmark; the 1.75% figure is for the 629-image subset. The gate is **not** a gradability assessment: it measures feature-space distance, not media opacity, pupil size, focus, or field placement. It has never been evaluated as an enforced input filter, and its ability to catch any specific failure mode is untested for this module.

---

## 10. Deployment artifacts

| Item | Value |
|---|---|
| ONNX file size | 212,474,067 bytes (approximately 203 MiB), measured on disk |
| PyTorch checkpoint | 214,269,117 bytes |
| OOD gate archive | 13,543,724 bytes |
| Operating-point configuration | approximately 5.5 KiB |
| Opset / input | 17 / (batch, 3, 480, 480), dynamic batch |
| Graph check | PASSED |
| Probability parity | maximum difference 0.000626 — PASSED |
| **Logit parity** | **maximum difference 0.002526 against a 0.001 tolerance — FAILED** |
| Verification samples | 3 batches x 2 images = 6 samples |
| Released asset (v1.0.0) | `retguard-dr-v1.0.0.zip`: `retguard_dr_v1.0.0.onnx` (the audited export, renamed, with CC BY-NC 4.0 metadata embedded), `ood_gate_dr_v1.0.0.npz`, `LICENSE.txt`, this model card |
| SHA-256, `retguard_dr_v1.0.0.onnx` | `f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b` |
| SHA-256, `ood_gate_dr_v1.0.0.npz` | `cb99b4d375ecc384ce6fbeff4f9b94c42613964b32b7a96cef8b3e1227dbf5b2` |
| Integrity | The zip digest and every member digest are published in the v1.0.0 release's `SHA256SUMS.txt` and embedded in `retguard/weights.py`; `retguard verify` re-checks them |

**The ONNX export failed its logit-parity tolerance.** Because deployment decisions consume calibrated probabilities, the practical impact is bounded by the probability difference (0.000626), but the check failed as recorded and export re-verification is required before any release. Six verification samples is a small parity check.

**Latency.** 11.9 ms at batch size 1; 2.9 ms per image at batch 4; 1.6 ms per image at batch 8. **These are PyTorch BF16 benchmark timings from the pipeline log, not ONNX Runtime timings, and the GPU model is not recorded.** No edge-hardware or ONNX Runtime benchmark has been run. No network connectivity is required at inference time.

---

## 11. Ethical considerations

1. **Low-prevalence PPV behavior.** All reported PPV values were measured on disease-enriched research cohorts. On the primary external benchmark, referable prevalence is 26.2% and PPV is 0.5904 — meaning roughly two of every five positive outputs were false referrals even at that elevated prevalence. In a real screening population with lower prevalence, PPV falls further and the false-referral count per true case rises. Any deployment claim about positive predictive value in a screening population would require prevalence-standardized analysis, which has not been performed.

2. **What a negative result means, and does not mean.** A non-referable output means only that this module did not detect referable diabetic retinopathy in this single image, under a threshold selected for high sensitivity on a different population. It is **not** a statement that the eye is healthy. In particular, the fundus source labels encode DR severity only, so **a non-referable output does not exclude diabetic macular edema**, which can occur at mild retinopathy grades. It also does not address glaucoma, AMD, or any other pathology. NPV on the external benchmark (0.9869) was measured at 26.2% prevalence and is not transferable.

3. **Referral-pathway dependence.** The output is a probability and a binary flag. It has clinical meaning only inside a defined referral pathway with a named clinician responsible for the decision, a defined route for flagged and for ungradable images, and a defined action on a negative result. No such pathway is specified, validated, or supplied with this module. Deploying a screening classifier without a functioning referral pathway can cause harm by generating referrals that cannot be absorbed, or by substituting for follow-up that then does not occur.

4. **Risk of use outside the intended input type.** This is a full-fundus-photograph classifier. Applying it to optic-disc crops, aggressively cropped fields, OCT, or other modalities is out of intended use. The companion glaucoma module, which shares this architecture and training discipline, performs **below chance** on optic-disc crops (AUC 0.3667), which demonstrates concretely that a RETGUARD fundus classifier can produce confident, systematically wrong output on an input type it was not built for. The advisory OOD gate has not been validated as a defense against this, for this module or any other.

5. **Population gaps as an equity issue.** African populations are essentially absent from training and evaluation. The screening gap that motivates this work concentrates in exactly the settings least represented in the data. Reporting a strong benchmark number while that gap is unmeasured risks overstating who the system has been shown to work for.

6. **Automation bias.** A calibrated probability presented to a non-specialist reader can anchor the reader's judgment. No human-factors evaluation of how this output affects grader behavior has been conducted.

---

## 12. Limitations and failure modes

Stated bluntly, and drawn from the manuscript's Limitations section and the DR verification manifest.

1. **Calibration exceeds the project's own gate.** In-sample calibration-set ECE is **0.052537**, above the project's 0.05 deployment gate. It is not a borderline pass; it is a fail on the module's own criterion, measured on the most favorable partition available.

2. **Calibration degrades under distribution shift.** Messidor-2 ECE is **0.0811**, against an in-sample calibration-set ECE of 0.052537 and a project gate of 0.05. IDRiD-Test ECE is 0.118801. The probabilities this module emits are not trustworthy off-domain, and no local recalibration procedure has been validated.

3. **The three post-hoc operating points are circular.** OP-1, OP-2, and OP-3 were derived on the Messidor-2 benchmark and their sensitivity/specificity are reported on that same benchmark — in-sample threshold selection. They carry **no portable performance claim** and are subordinate to the pre-specified-threshold result (sensitivity 0.9716 / specificity 0.7607). Any use of OP-1's 0.9103/0.8990 as a deployment expectation is unsupported.

4. **ONNX export logit-parity tolerance failure.** Maximum logit difference 0.002526 against a 0.001 threshold — recorded as FAILED. Probability parity passed at 0.000626 and the graph check passed, so practical impact is bounded, but the export is unverified as released. Re-verification is required.

5. **The deployed preprocessing recipe is not evidenced by run artifacts; it is reconstructed from source.** The preprocessing phase was skipped in the audited pipeline run (pre-existing processed data), so the DR module's input-preprocessing specification is **not evidenced by artifacts of that run**. It has since been fully reconstructed from the released source code: byte-for-byte the same five-step LOCKED fundus function as the glaucoma module (paper Section 2.2, verified from released source), and the accompanying inference package ships it parity-verified. The residual limitation is that the recipe's evidence is the released source, not an artifact of the executed run.

6. **No Venn-Abers layer.** This module uses temperature scaling only. It lacks the distribution-free validity guarantee applied in the glaucoma and OCT modules.

7. **Case definitions are under-specified.** The exact referable-DR grade cutoff, the sight-threatening grade boundary, and the Messidor-2 label provenance are not recorded in the audited artifacts. Every sight-threatening-sensitivity figure in Section 7.1 therefore rests on an undocumented boundary.

8. **DME is invisible to this module.** The source labels encode DR severity only. A non-referable output does not exclude diabetic macular edema, whereas the reference standards of cleared comparator systems include DME.

9. **The dataset-family-holdout rows are not external validation.** APTOS, IDRiD, and DeepDRiD evaluated splits come from datasets whose train splits contributed to training. Split verification recorded 0 overlapping images and 0 overlapping patients, but the external-evaluation artifact itself warns that IDRiD and DeepDRiD metrics "may be inflated." The only strictly zero-shot external evidence for this module is Messidor-2.

10. **A stale Messidor-2 split file exists and contradicts the exclusion claim on its face.** `verification_report.json` contains a Messidor-2 train/eval split (train 504, eval 125 of 629). The pipeline log confirms Messidor-2 was excluded from training for this run, and the split file predates the run, but a reader of that artifact alone would conclude otherwise. The file must be removed or annotated in the released repository.

11. **Small-sample and wide-CI evaluations.** IDRiD-Test (n=71, 26 negatives) and DeepDRiD-Val (n=302) and APTOS (n=461) carry small-sample warnings; IDRiD-Test additionally carries a wide-CI warning. The 0.0501 best-to-worst AUC spread flagged by the pipeline is driven by these small sets.

12. **Per-dataset CIs use 1,000 resamples, below the project's own convention.** The bootstrap protocol is recovered from the released source (percentile type, unstratified, fixed seeds; paper Section 2.7): per-dataset AUC/sensitivity/specificity CIs use 1,000 resamples (seeds 42/43/44) while operating-point CIs use 2,000. The 1,000-resample count falls short of the project's 2,000-resample reporting standard, and the counts are recorded in source rather than in the evaluation artifacts.

13. **No split-verification artifact for EyePACS or DDR.** The evaluation sample sizes are confirmed, but no artifact verifies that the EyePACS and DDR train/validation/test partitions are patient-disjoint. If same-patient images straddle those partitions, the internal AUCs (0.9475, 0.9549) and the calibration estimates fitted on the EyePACS calibration partition are optimistically biased.

14. **Single training run, single seed.** Run-to-run variance is unquantified; no repeat-run artifact exists. No claimed variance bound is supported.

15. **Threshold prevalence sensitivity.** The selection rule (maximize specificity subject to sensitivity >= 0.90) is recovered from the released source, not from run artifacts (paper Section 2.4). The threshold was selected on a 2,358-image EyePACS partition; the bootstrap specificity CI on that partition is wide (0.7708-0.8839). At a different deployment prevalence, the achieved sensitivity/specificity trade-off will shift.

16. **Per-image, not per-patient.** All operating characteristics are per-image. Clinical screening decisions are per-patient and typically integrate multiple fields per eye. No per-patient aggregation rule is defined or validated.

17. **No gradability or imageability pathway.** All benchmarks are curated and gradability-filtered. Prospective imageability — the metric on which cleared systems report 87-99% — is unmeasured. No mydriasis, field, or resolution input specification is defined. Deployment would require an explicit ungradable pathway distinct from the OOD gate.

18. **Retrospective only.** Every result is retrospective and computed on curated public datasets. No prospective, intent-to-screen, or real-device validation exists. The documented retrospective-to-prospective compression in this field should be assumed to apply: the IDx-DR lineage reported 96.8% sensitivity retrospectively on Messidor-2 but 87.2% in its prospective pivotal trial.

19. **Reference-standard limitations.** Labels are the public datasets' labels after confident-learning cleaning. The exact Messidor-2 grade-set provenance is undocumented, and published comparators used expert-regrade reference standards, which biases any comparison in an unknowable direction.

20. **No regulatory status.** RETGUARD has no regulatory clearance, authorization, or certification anywhere. It is not a medical device.
