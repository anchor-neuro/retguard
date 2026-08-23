# Model Card — RETGUARD Glaucoma Module (Fundus)

**Model card version:** 1.0 | **Date:** 2026-08-22 | **Status:** Research software. No regulatory clearance.

> **This is research software. It is not a medical device. It has no FDA clearance, no CE mark, and no regulatory authorization in any jurisdiction. It must not be used to make or inform any clinical decision about any patient.**
>
> **This module performs below chance on optic-disc crops (AUC 0.3667). It is a full-fundus-photograph classifier only. See Sections 2, 7.2, and 12.**

This card follows the Model Cards for Model Reporting framework (Mitchell M, Wu S, Zaldivar A, et al. Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*. 2019:220-229. DOI: 10.1145/3287560.3287596).

Every number in this card is copied verbatim from the RETGUARD manuscript (v1.2, 2026-08-22) or from the independent verification manifest `manifest_Glaucoma.json` (generated 2026-08-22). Nothing is estimated, and nothing is rounded beyond the precision the source artifact stores. Values the audit could not trace to an artifact are labeled **report-stated** wherever they appear.

---

## 1. Model details

| Item | Value |
|---|---|
| Developer | Sameh Aboelmaaty, Anchor Neuro, Delaware, USA |
| Model card date | 2026-08-22 |
| Module version | RetGuard v6 Binary Glaucoma Classifier, EfficientNetV2-M, champion checkpoint epoch 26 (benchmark artifacts dated 2026-03-10) |
| Model type | Binary image classifier (referable glaucoma vs. non-referable) |
| Architecture | EfficientNetV2-M, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k; export metadata records only "efficientnetv2_m + full fine-tune (eval mode)" |
| Input | Single color **full fundus photograph**, 480 x 480 x 3 |
| Trainable parameters | 53,186,549, all trainable (pipeline log; paper Section 2.3). The module report is internally contradictory on this point (53.2M in one section, 54.1M in another); the pipeline-log value governs |
| Output | Calibrated probability of referable glaucoma; binary decision at the fixed threshold 0.044776 |
| Calibration | Temperature scaling (temperature 0.94653) plus an Inductive Venn-Abers Predictor (IVAP); deployment calibration flag is `venn_abers`, with temperature scaling as fallback |
| Test-time augmentation | 8 views (D4: 4 rotations x 2 flips) |
| Out-of-distribution gate | Mahalanobis-distance detector on penultimate features, 97th-percentile threshold 48.3972; advisory only |
| Export format | ONNX, opset 17, dynamic batch, input 480; TensorRT precision field FP16; file size 212,474,067 bytes measured on disk |
| Companion artifacts | Venn-Abers calibrator 43,446 bytes; OOD gate archive 13,796,684 bytes; PyTorch checkpoint 214,331,421 bytes |
| Training seed | 42, all stages (verified from released source; single run) |
| License — weights | CC BY-NC 4.0, research use; commercial licensing available separately (see LICENSE-WEIGHTS.md, COMMERCIAL-LICENSE.md) |
| License — code | PolyForm Noncommercial 1.0.0 (see LICENSE.md) |
| Contact | ghoneim2012@gmail.com |
| Repository | https://github.com/anchor-neuro/retguard |
| Weights mirror | https://huggingface.co/anchor-neuro/retguard |

**Paper citation.** Aboelmaaty S. RETGUARD: Calibrated, Out-of-Distribution-Aware Deep Learning for Multi-Disease Retinal Screening from Fundus Photography and Optical Coherence Tomography, with External Validation and Explicit Failure-Mode Reporting. Manuscript v1.2, 2026-08-22. Anchor Neuro, Delaware, USA. Submission-ready draft (medRxiv).

**Required input preprocessing (recorded in export metadata, deployment-relevant, order is load-bearing):**

1. Circle crop
2. Resize to 480 px (Lanczos4)
3. Ben Graham unsharp masking: 4\*img − 4\*blur + 128, sigma 16.0
4. CLAHE on the LAB L-channel, clip limit 2.0, 8x8 grid
5. Circular mask, radius 216 px
6. Normalization: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], RGB color space

The export metadata records `ben_graham_required_upstream: true` and `clahe_required_upstream: true`. Note that the module report states this order incorrectly (it lists CLAHE before unsharp masking and omits the resize step); the artifact order above is authoritative.

**Verified training configuration:** 34 epochs run; champion epoch 26 with validation AUC 0.9834748584748585 (maximum across all 34 epochs); per-epoch losses, AUC, sensitivity, specificity, F1, Brier, threshold, backbone learning rate, gradient-norm mean, and sampling statistics are recorded. Epoch-1 backbone learning rate 5.00374531835206e-05.

**Configuration items not recorded in any evaluation artifact, subsequently recovered from the released source code by an independent audit (paper Table 2, cells marked *(src)*):** AdamW with backbone learning rate 1e-4 and head 3e-4 (resolving the module reports' 3e-4-vs-1e-4 contradiction), weight decay 1e-4, linear warmup then cosine decay, focal loss (gamma 2.0) with a MixUp mixture, batch size 16 with 4-step gradient accumulation, and BF16 precision. The parameter count is in the pipeline log (above); the bootstrap protocol is specified in paper Section 2.7 (1,000 resamples for per-dataset metric CIs, 2,000 for operating-point CIs). The GPU hardware of the executed run remains unrecorded.

---

## 2. Intended use

**Primary intended use.** Research and methodological evaluation only: retrospective benchmarking of referable-glaucoma classification from full fundus photographs, calibration behavior, out-of-distribution flagging, preprocessing-domain sensitivity, and operating-point portability on public datasets.

**Primary intended users.** Machine-learning researchers and clinical-AI evaluators working with de-identified, publicly available retinal image datasets.

**Required input type.** A single color **full fundus photograph** of the posterior pole, preprocessed exactly as specified in Section 1, resized to 480 x 480.

**Out of scope — this module must not be used for:**

- Any clinical decision, triage, referral, diagnosis, or exclusion of glaucoma for any patient.
- Any use implying regulatory clearance. There is none, in any jurisdiction.
- **Optic-disc crops.** Measured performance on RIM-ONE DL optic-disc crops is AUC 0.3667 — below chance. This is a hard intended-use boundary, not a caution.
- **Aggressively cropped fields of view.** Measured performance on ACRIMA is AUC 0.7185 (minimal preprocessing) and 0.6049 (training-domain preprocessing).
- Any imaging type other than a color fundus photograph — not OCT, not RNFL scans, not visual-field data.
- Confirming or excluding glaucoma. A fundus photograph carries no intraocular pressure, no visual-field, and no RNFL OCT information. Photograph-based glaucoma detection has an intrinsic ceiling, and a negative output does not exclude glaucoma.
- Any autonomous or unattended operation. There is no gradability/imageability pathway.
- Deployment on any device or in any population without prospective, target-hardware validation and local recalibration.

---

## 3. Factors

**Populations and acquisition settings represented in training:**

| Dataset | Origin / device | Role |
|---|---|---|
| AIROGS | Screening program, approximately 500 centers | Train + internal evaluation (75,149 post-cleaning) |
| G1020 | Germany, private clinical practice, Kaiserslautern; 432 patients | Auxiliary train (retained remainder) + 155-image held-out evaluation. Note: one artifact describes G1020 as a "Pakistani population" cohort while the manuscript's dataset table records a German clinical practice; this provenance conflict is unresolved |
| DRISHTI-GS (train split) | India | Auxiliary train (81 images; verified from the pipeline execution log) |
| FIVES (train split) | China | Auxiliary train (727 images; verified from the pipeline execution log) |
| PAPILA, sjchoi86-HRF, ODIR-5K, HRF, Chaksu (Indian population) | Various | Auxiliary train (342 / 630 / 6,547 / 30 / 1,319 images; verified from the pipeline execution log) |

**Populations represented in evaluation:** REFUGE (challenge validation set), ORIGA (Singapore, Singapore Malay Eye Study), DRISHTI-GS (India), FIVES (China), G1020 (see conflict above), RIM-ONE DL (3 Spanish hospitals, optic-disc crops), ACRIMA (cropped field of view).

**The auxiliary training ledger is now itemized from the pipeline execution log** (paper Section 2.1): nine of eleven candidate datasets entered training — AIROGS, G1020, FIVES, PAPILA, sjchoi86-HRF, ODIR-5K, HRF, Chaksu, and DRISHTI-GS — a pooled corpus of 85,448 images (2,881 referable / 82,567 non-referable), with BEH and CRFO absent from disk and excluded. This supersedes the module report's auxiliary list, which **omits G1020 despite its confirmed use**. Two residual weaknesses remain: the training code never persists a dataset manifest (the composition is recoverable only from captured standard output), and **no image-hash overlap check between the training pool and any external benchmark has been run.** Every "zero-shot" label in this card is conditional on that pending overlap check.

**Known geographic and ethnic gaps.** Training and evaluation data derive predominantly from European (including German and Spanish), Chinese, Indian, and Singaporean cohorts. **African populations are essentially absent from both training and evaluation.** Performance in those populations is unknown, not merely unproven. No per-subgroup analysis by age, sex, ethnicity, or camera model was performed.

**Image-type factor dominates every other factor measured.** The single largest performance determinant identified is whether the input is a full fundus photograph or a crop. That factor spans AUC 0.9988 to 0.3667 across the benchmarks in this card.

**Reference-standard heterogeneity is a factor, not a detail.** ORIGA labels are substantially cup-disc-ratio-derived; FIVES "glaucoma" is a coarse diagnosis category in a vessel-segmentation dataset whose negatives include AMD and DR eyes; DRISHTI-GS is an optic-nerve-head segmentation dataset with 74.5% glaucoma prevalence in its test split; AIROGS uses grader-assessed fundus signs of glaucomatous optic neuropathy warranting referral. Cross-dataset results therefore conflate structural suspicion with clinical diagnosis to differing degrees.

---

## 4. Metrics

| Metric | Why it is reported |
|---|---|
| AUC with 95% bootstrap CI | Threshold-free discrimination; the only portable comparator across datasets with differing prevalence and reference standards |
| Sensitivity and specificity at the fixed threshold (0.044776), with 95% bootstrap CIs | The deployment-relevant operating characteristic at a threshold fixed before any benchmark was seen and never tuned per benchmark |
| Sensitivity at 90% and 95% specificity | Comparison against the AIROGS challenge's screening convention and published zero-shot models |
| PPV, NPV, F1, full confusion matrix | Referral-workload interpretation. PPV and NPV are prevalence-dependent and are reported with the cohort prevalence stated |
| ECE, MCE, Brier score | Calibration is a safety property. Reported separately for **in-sample** assessment (the calibrator's own fitting partition, optimistically biased by construction) and **held-out** assessment |
| Venn-Abers interval width (mean, median, maximum, SD) | The IVAP emits a probability interval with a distribution-free validity guarantee; interval width signals per-input uncertainty |
| OOD flag rate per dataset | Measures how often the advisory gate would route an input to human review. Reported as a result, never tuned |

**Statistical protocol.** All CIs are bootstrap-based, of the percentile type, unstratified, with fixed seeds (verified from released source; paper Section 2.7). The operating-point threshold, sensitivity, and specificity CIs use 2,000 resamples (seed 42); every per-dataset AUC, sensitivity, and specificity CI uses 1,000 resamples (seeds 42, 43, and 44 respectively) — below the project's own 2,000-resample reporting convention (Section 12, item 15). ECE was computed with 15 bins where the bin count is recorded (the temperature-scaling block). Bootstrap CIs reported as [1.0000, 1.0000] arise from resampling perfectly separated data; they are degenerate at the parameter boundary, understate sampling uncertainty, and are excluded from any pooled-mean inference.

---

## 5. Training data

**AIROGS, after confident-learning label cleaning:**

| Item | Value |
|---|---|
| Initial n | 79,538 |
| Final n | **75,149** |
| Excluded as uncertain | 4,389 (5.52%) |
| Initial class distribution | non-referable 77,213 / referable 2,325 |
| Final class distribution | non-referable 73,430 / referable 1,719 |
| Initial data-quality score | 0.9352762201714904 |
| Data-quality score after cleaning | 0.9899000652037951 |
| Cleaning version | v6-binary-single-pass |

**G1020, after confident-learning label cleaning:**

| Item | Value |
|---|---|
| Images processed | 1,017 (the published dataset has 1,020; the 3-image difference is undocumented) |
| Excluded as uncertain | 239 (23.5%) |
| Retained | 778 |
| Held out for evaluation | 155 (20%, patient-stratified) |
| Remainder | Used as auxiliary training data |
| Initial data-quality score | **0.6843657817109144** (versus 0.9352762201714904 for AIROGS) |

Whether the 155-image evaluation split itself passed through label cleaning is not recorded.

**Partitions:** train 75,149 (AIROGS, post-cleaning) plus auxiliary sets; validation holdout 8,837; threshold/calibration partition 2,651.

**Auxiliary training data beyond AIROGS and G1020** is itemized in the pipeline execution log (paper Section 2.1): FIVES 727 (188 referable / 539 non-referable), PAPILA 342 (96/246), sjchoi86-HRF 630 (158/472), ODIR-5K 6,547 (375/6,172), HRF 30 (15/15), Chaksu 1,319 (160/1,159), and DRISHTI-GS 81 (55/26) — which, together with AIROGS (75,149) and the G1020 retained remainder (623; 115/508), gives a pooled corpus of 85,448 images (2,881/82,567) across nine datasets. The composition exists only in captured standard output — the training code never serializes a dataset manifest to disk, so a rerun on a machine holding the absent BEH and CRFO datasets would silently train on a different corpus.

**Case definition.** Referable glaucoma follows the AIROGS RG label: grader-assessed fundus signs of glaucomatous optic neuropathy warranting referral, per the challenge grading protocol.

**License and access status of the datasets used:**

| Dataset | License / access terms |
|---|---|
| AIROGS | Challenge terms |
| G1020 | Research use |
| DRISHTI-GS | Research use |
| FIVES | Research use |
| REFUGE | Challenge terms |
| ORIGA | Research use |
| RIM-ONE DL | Research use |
| ACRIMA | Research use |

None of these access terms was independently verified against the source repositories by the audit. **Commercial use of a model trained on these datasets is not established as permitted and must be reviewed against each dataset's terms before any such use.**

---

## 6. Evaluation data

Classification uses exactly the manuscript's terms. A dataset is **external (zero-shot)** only if no image or sibling split of that dataset contributed to training. A dataset whose train split contributed to training, but whose evaluated split is disjoint, is a **dataset-family holdout** — this is *not* external validation.

| Evaluation set | n (pos/neg) | Image type | Classification (manuscript's exact term) |
|---|---|---|---|
| AIROGS-Val | 8,837 (258 / 8,579); prevalence 2.92% | full fundus | Internal held-out partition |
| Threshold/calibration partition | 2,651 | full fundus | In-sample for the fitted calibrators |
| REFUGE-Val | 400 (40 / 360); prevalence 0.10 | full fundus | **External (zero-shot;** training corpus itemized in Section 3; conditional on the pending image-hash overlap check**)** |
| ORIGA | 650 (168 / 482); prevalence 0.2585 | full fundus | **External (zero-shot;** training corpus itemized in Section 3; conditional on the pending image-hash overlap check**)** |
| DRISHTI-GS-Test | 51 (38 / 13); prevalence 0.7451 | full fundus | **Dataset-family holdout** (train split used as auxiliary training data; verified from execution log) |
| FIVES-Test | 200 (50 / 150); prevalence 0.25 | full fundus | **Dataset-family holdout** (train split used as auxiliary training data; verified from execution log) |
| G1020-Eval | 155 (29 / 126); prevalence 0.1871 | full fundus | **Dataset-family holdout** (retained remainder used as auxiliary training data) |
| RIM-ONE DL | 485 (172 / 313); prevalence 0.3546 | **optic-disc crop** | External (zero-shot); **out of intended use** |
| ACRIMA | 705 (396 / 309); prevalence 0.5617 | **cropped field of view** | External (zero-shot); **out of intended use** |

**Only REFUGE-Val and ORIGA are strictly zero-shot full-fundus benchmarks.** DRISHTI-GS and FIVES are dataset-family holdouts; their near-ceiling results are not external validation and must never be presented as such.

**Two evaluation protocols were run on all six benchmarks, and both are reported in full:**

- **Minimal:** resize to 480 x 480 plus ImageNet normalization only.
- **Ben Graham:** the full five-step training-domain pipeline of Section 1.

Both protocols used the same fixed threshold 0.044776, 8-view TTA, Venn-Abers calibration, and no benchmark-specific tuning.

---

## 7. Quantitative results

### 7.1 Internal held-out evaluation (AIROGS-Val, n = 8,837)

| Quantity | Value |
|---|---|
| AUC (95% CI) | **0.9792 (0.9721-0.9851)** |
| Sensitivity (95% CI) | 0.9302 (0.8972-0.9595) |
| Specificity (95% CI) | 0.9399 (0.9346-0.9447) |
| Confusion matrix | TP 240, TN 8,063, FP 516, FN 18 |
| PPV / NPV / F1 | 0.3175 / 0.9978 / 0.4734 |
| Sensitivity at 90% specificity | 0.9535 |
| Sensitivity at 95% specificity | 0.8798 |
| Brier | 0.013532 |
| ECE (held-out with respect to the calibrator's fitting partition) | 0.003616 |
| Prevalence | 2.92% |
| TTA | 8 views |

The PPV of 0.3175 reflects the 2.92% prevalence: roughly two of every three positive outputs were false positives at that prevalence. PPV and NPV are prevalence-dependent.

For context under matched conventions, the best AIROGS challenge team reported sensitivity 0.85 (95% CI 0.83-0.87) at 95% specificity on the *hidden* challenge test set, with a human grader panel at sensitivity 0.86 / specificity 0.94. RETGUARD's 0.8798 at 95% specificity is on the public validation partition — a different evaluation set — so the comparison is indicative only.

### 7.2 External and dataset-family-holdout benchmarks, both protocols

Fixed threshold 0.044776, 8-view TTA, no benchmark-specific tuning.

| Dataset (n; pos/neg) | Image type | Exposure | AUC minimal (95% CI) | AUC Ben Graham (95% CI) | Sens/Spec minimal | Sens/Spec Ben Graham |
|---|---|---|---|---|---|---|
| REFUGE-Val (400; 40/360) | full fundus | external (zero-shot) | 0.9278 (0.8642-0.9763) | 0.9119 (0.8420-0.9703) | 0.7000 / 0.9944 | 0.7250 / 0.9917 |
| ORIGA (650; 168/482) | full fundus | external (zero-shot; see Section 3 caveat) | 0.8729 (0.8399-0.9028) | **0.9988 (0.9964-1.0000)** | 0.5893 / 0.9253 | 0.9940 / 0.9813 |
| FIVES-Test (200; 50/150) | full fundus | **dataset-family holdout** | 0.8317 (0.7506-0.8974) | 1.0000 (1.0000-1.0000)\* | 0.6200 / 0.8667 | 1.0000 / 0.9933 |
| DRISHTI-GS-Test (51; 38/13) | full fundus | **dataset-family holdout** | 0.9615 (0.9073-0.9942) | 1.0000 (1.0000-1.0000)\* | 1.0000 / 0.5385 | 1.0000 / 0.7692 |
| **RIM-ONE DL (485; 172/313)** | **optic-disc crop** | external; **out of intended use** | **0.3667 (0.3145-0.4164)** | **0.4735 (0.4209-0.5264)** | 0.1802 / 0.7508 | 0.4012 / 0.5463 |
| **ACRIMA (705; 396/309)** | **cropped FOV** | external; **out of intended use** | **0.7185 (0.6824-0.7529)** | **0.6049 (0.5648-0.6465)** | 0.3611 / 0.9061 | 0.5202 / 0.6472 |
| Mean, 2 strictly zero-shot full-fundus sets (computed) | | | 0.9004 | 0.9554 | | |
| Mean, 4 full-fundus sets (2 zero-shot + 2 family holdout) (computed) | | | 0.8985 | 0.9777 | | |
| Mean, all 6 (computed) | | | 0.7799 | 0.8315 | | |

\* Bootstrap CI degenerate at the parameter boundary under perfect separation (every resample reproduces AUC 1.0). The interval understates sampling uncertainty (n=51 for DRISHTI-GS, with only 13 negatives). These rows are excluded from pooled-mean inference.

Additional stored calibration values, minimal protocol: REFUGE-Val Brier 0.065091 / ECE 0.077171; ORIGA Brier 0.211385 / ECE 0.222428; FIVES-Test Brier 0.159607 / ECE 0.177352; DRISHTI-GS-Test Brier 0.255096 / ECE 0.401042; RIM-ONE DL Brier 0.343068 / ECE 0.333024; **ACRIMA Brier 0.511797 / ECE 0.527851**.

Additional stored calibration values, Ben Graham protocol: REFUGE-Val Brier 0.054387 / ECE 0.064377; ORIGA Brier 0.005188 / ECE 0.014981; FIVES-Test Brier 0.004246 / ECE 0.020969; DRISHTI-GS-Test Brier 0.009319 / ECE 0.05732; RIM-ONE DL Brier 0.319078 / ECE 0.280844; ACRIMA Brier 0.445357 / ECE 0.444048.

### 7.3 The ORIGA 0.9988 result is unconfirmed pending replication

**The Ben Graham ORIGA AUC of 0.9988 exceeds every published ORIGA classification figure the manuscript could verify.** The trained-on-ORIGA ceiling in the literature is 0.831-0.851, and well-executed zero-shot models report 0.85-0.854. This value is reported here as stored in the artifacts and is **treated as unconfirmed until independently replicated**. Two possibilities cannot be formally excluded:

1. **Undetected training exposure.** The training corpus is now itemized from the execution log (Section 3) and ORIGA is not among the nine pooled datasets, but no image-hash overlap check has been run, so undetected duplication of ORIGA or related Singapore Malay Eye Study imagery within another pooled source cannot be formally excluded.
2. **An undetected evaluation artifact** in the benchmark pipeline.

The minimal-protocol ORIGA result, 0.8729 (0.8399-0.9028), is consistent with the published zero-shot range and is the more defensible of the two figures.

### 7.4 Effect of training-domain preprocessing — a hypothesis, not an established effect

Per-dataset AUC deltas (Ben Graham minus minimal), from the stored comparison block:

| Dataset | AUC delta | Sensitivity delta | Specificity delta |
|---|---|---|---|
| REFUGE-Val | **−0.0159** | +0.0250 | −0.0027 |
| ORIGA | +0.1259 | +0.4047 | +0.0560 |
| FIVES-Test | +0.1683 | +0.3800 | +0.1266 |
| DRISHTI-GS-Test | +0.0385 | 0.0000 | +0.2307 |
| RIM-ONE DL | +0.1068 | +0.2210 | −0.2045 |
| ACRIMA | **−0.1136** | +0.1591 | −0.2589 |

Aggregates over the four full-fundus sets: mean AUC rose by 0.0792 (0.8985 to 0.9777); mean Brier fell from 0.172795 to 0.018285; mean ECE fell from 0.219498 to 0.039412. The internal-to-external mean-AUC difference was 0.0807 (minimal) and 0.0015 (Ben Graham).

**These aggregates are driven by the two dataset-family holdouts reaching a degenerate 1.0000 and by the unconfirmed ORIGA result.** On the only benchmark that is both strictly zero-shot and unflagged — REFUGE-Val — Ben Graham preprocessing *decreased* AUC, and it also decreased AUC on ACRIMA. A dataset-family holdout reaching exactly 1.0 after training-domain preprocessing is also consistent with the model recognizing sibling-split dataset statistics rather than with genuine domain harmonization. The effect is therefore stated as a hypothesis pending replication, not as an established first-order effect. **No per-delta CIs were computed;** paired bootstrap CIs on the deltas are required.

### 7.5 G1020 dataset-family holdout — near-chance, unresolved

Two independent runs exist on the same 155-image split, and both are reported:

| Run | Protocol | AUC (95% CI) | Sens. | Spec. | PPV | NPV | Confusion (TP/TN/FP/FN) | Brier | ECE | Sens@90spec | Sens@95spec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Evaluation pipeline | minimal | 0.5896 (0.4792-0.6975) | 0.4138 | 0.7937 | 0.3158 | 0.8547 | 12/100/26/17 | 0.180165 | 0.180652 | 0.1724 | 0.0345 |
| Benchmark run | Ben Graham | 0.5813 (0.4726-0.6846) | 0.3793 | 0.7778 | 0.2821 | 0.8448 | 11/98/28/18 | 0.185304 | 0.170446 | 0.1483 | 0.0345 |

Both are near chance with wide CIs. Two artifact-supported context points: G1020's confident-learning initial data-quality score was 0.6843657817109144 (versus 0.9352762201714904 for AIROGS), with 23.5% of images flagged; and the OOD gate flagged 28.39% of G1020 inputs versus 4.99% of AIROGS-Val. **This is reported as an unresolved poor result on a small, label-noisy, distribution-shifted split, not attributed to any single cause.** The internal-to-G1020 AUC gap of 0.3896 is flagged as a warning by the evaluation pipeline itself.

### 7.6 Operating point

| Item | Value |
|---|---|
| Selection method | Maximize specificity subject to sensitivity >= 0.90 (the target is not stored in the artifacts; the rule and its 0.90 constraint were recovered from the released source code — paper Section 2.4). The module reports characterize the threshold as "Youden's J," which is a different criterion and is contradicted by both artifact and source |
| Threshold | 0.044776 |
| Sensitivity / specificity on the threshold partition | 0.9103 / 0.9390 |
| AUC on the threshold partition (95% CI) | 0.974033 (0.9576-0.9861) |
| Bootstrap (2,000 valid resamples) | threshold mean 0.062894, SD 0.061136; sensitivity CI 0.9000-0.9867; **specificity CI 0.7172-0.9724** |

The wide bootstrap specificity CI at this fixed low threshold reflects the low-prevalence (2.92%) calibration partition and is a portability caution: at a different deployment prevalence, the achieved sensitivity/specificity trade-off will shift materially.

### 7.7 Release gate and explainability inventory

The automated release gate passed (`kill_gate_passed = true`). **The numeric gate threshold value is not stored in any artifact.** Grad-CAM panels were generated for 50 images (10 referable, 40 non-referable).

---

## 8. Calibration

| Item | Value |
|---|---|
| Calibrator types | Temperature scaling, plus an Inductive Venn-Abers Predictor (IVAP, Vovk & Petej 2014) |
| Temperature | 0.94653 |
| Fitting partition | 2,651 AIROGS images |
| Deployment calibration flag | `venn_abers` (primary), `temperature_scaling` (fallback) |
| ECE bins | 15 (temperature-scaling block) |

**In-sample assessment** — computed on the same 2,651-image partition used to fit the calibrators, and therefore **optimistically biased by construction**:

| Calibrator | Brier | ECE | MCE |
|---|---|---|---|
| Temperature scaling | 0.015845 | 0.008543 | 0.398333 |
| Venn-Abers (IVAP) | 0.014258 | **0.001493** | 0.809524 |

The in-sample ECE ratio is 0.008543 / 0.001493 = 5.72x. **For isotonic-family calibrators such as IVAP, near-zero in-sample ECE is close to guaranteed by construction.** The 0.001493 figure is a fitting diagnostic, not evidence of deployed calibration quality, and must never be quoted as a deployment calibration result.

Venn-Abers interval widths on the fitting partition: mean 0.000747, median 0.0, **maximum 0.264706**, SD 0.008809. The maximum interval width is informative: for a minority of inputs the calibrator itself signals substantial uncertainty.

**Held-out assessment** — the honest post-calibration estimate:

| Partition | ECE | Brier |
|---|---|---|
| **AIROGS-Val (held-out)** | **0.003616** | 0.013532 |

**External calibration under distribution shift is far worse.** Mean ECE across the four full-fundus benchmarks was **0.219498** under the minimal protocol and 0.039412 under the Ben Graham protocol. On ACRIMA under the minimal protocol, ECE reached **0.527851** and Brier 0.511797 — calibration collapses in the same direction as discrimination on out-of-intended-use inputs.

**Interval-to-scalar rule.** The Venn-Abers interval is reduced to the scalar probability the fixed threshold consumes by the Vovk regularized point estimate, p = p1 / (1 − p0 + p1), with the denominator floored at 1e-15 and the result clipped to [0, 1] — not the interval midpoint and not p1 alone (not recorded in the evaluation artifacts; verified from released source, paper Section 2.4, and implemented identically in the accompanying inference package). The Venn-Abers validity guarantee holds under i.i.d. sampling and does not transfer to shifted prevalence.

---

## 9. Out-of-distribution gate

| Item | Value |
|---|---|
| Method | Mahalanobis distance on penultimate backbone features |
| Feature dimension | 1,280 (no evaluation artifact states it directly; the dimension is hard-asserted in the released OOD-gate source code — paper Section 2.4 — and supported indirectly by the cleanlab probe architecture MLP(1280→512→2)) |
| Threshold percentile | 97th percentile of training-set scores |
| Threshold value | 48.3972 |
| Behavior | **Advisory only.** Flags inputs for human review; does not alter or suppress predictions |

**Flag rates — only two datasets have recorded OOD outputs for this module:**

| Dataset | Flagged | Flag rate | Mean score | Maximum score |
|---|---|---|---|---|
| AIROGS-Val | 441 | 4.99% | 32.3594 | 70.1156 |
| G1020-Eval | 44 | 28.39% | 44.7431 | 78.1122 |

**What was and was not tested — stated plainly.**

- **Tested:** flag rates on AIROGS-Val and G1020-Eval only.
- **Not tested:** the benchmark artifacts contain **no OOD fields for any of the six external benchmarks**. In particular, **the gate was never run on RIM-ONE DL or ACRIMA — the exact inputs on which this classifier fails.** Whether the gate would catch the below-chance optic-disc-crop failure mode is **unknown**. The proposed hard input filter is a design intention, not a validated mechanism.
- **Specified from source, not from artifacts:** the detector's exact form is absent from the evaluation artifacts but is now specified from the released source (paper Section 2.4) — a single pooled Gaussian over L2-normalized features, empirical covariance plus a 1e-5 ridge, Mahalanobis distance (not its square) via Cholesky forward substitution.
- **Not what it measures:** the gate is not a gradability assessment. It measures feature-space distance, not media opacity, pupil size, focus, or field placement.

Measuring OOD flag rates on RIM-ONE DL and ACRIMA is a prerequisite before the gate could be promoted from an advisory flag to an enforced input filter.

---

## 10. Deployment artifacts

| Item | Value |
|---|---|
| ONNX file size | 212,474,067 bytes (approximately 202.6 MiB), measured on disk |
| PyTorch checkpoint | 214,331,421 bytes |
| Venn-Abers calibrator | 43,446 bytes |
| OOD gate archive | 13,796,684 bytes (the module reports state "<1 MB", which is wrong by more than an order of magnitude) |
| Opset / input | 17 / 480 |
| TensorRT precision field | FP16 |
| Export numeric verification | Not recorded in the artifacts read by the audit for this module |
| Released asset (v1.0.0) | `retguard-glaucoma-v1.0.0.zip`: `retguard_glaucoma_v1.0.0.onnx` (the audited export, renamed, with CC BY-NC 4.0 metadata embedded), `ood_gate_glaucoma_v1.0.0.npz`, `venn_abers_glaucoma_v1.0.0.npz`, `LICENSE.txt`, this model card |
| SHA-256, `retguard_glaucoma_v1.0.0.onnx` | `b5686229ecc9050caddf61a66686e77e3e5cb4085425cca881078acb586cbb7b` |
| SHA-256, `ood_gate_glaucoma_v1.0.0.npz` | `5d07b5fbd8c0a77b8319e09f7e2b26b95ce1cc5b006705aec4864dd119ebde8d` |
| SHA-256, `venn_abers_glaucoma_v1.0.0.npz` | `8641e854024ae1ff2bdb8ecfdc6f6edb1a7dbf79674c9252c23f490437c5c405` |
| Integrity | The zip digest and every member digest are published in the v1.0.0 release's `SHA256SUMS.txt` and embedded in `retguard/weights.py`; `retguard verify` re-checks them |

The preprocessing pipeline of Section 1 must ship with the model, byte-exact. Section 7.4 shows that changing preprocessing moves AUC by up to 0.1683 on a single dataset and by −0.1136 in the opposite direction on another. No network connectivity is required at inference time. No ONNX Runtime or edge-hardware benchmark has been run for this module.

---

## 11. Ethical considerations

1. **Low-prevalence PPV behavior.** On the internal held-out partition at 2.92% prevalence, PPV is **0.3175** — approximately two of every three positive outputs are false positives. This is the expected behavior of a high-sensitivity threshold in a low-prevalence population, and it is what a real screening deployment would look like. Benchmarks with enriched prevalence report much better PPV (ORIGA at 25.85% prevalence: 0.7333 minimal, 0.9489 Ben Graham) and are not representative of screening. Any deployment claim about PPV would require prevalence-standardized analysis, which has not been performed.

2. **What a negative result means, and does not mean.** A non-referable output means only that this module did not detect referable glaucomatous optic neuropathy in this single fundus photograph. **Glaucoma cannot be confirmed or excluded from a fundus photograph** — no intraocular pressure, no visual fields, no RNFL OCT. A negative output does not exclude glaucoma, and must never be communicated as if it did. NPV on the internal partition (0.9978) was measured at 2.92% prevalence and reflects that prevalence, not diagnostic certainty.

3. **Referral-pathway dependence.** The output has clinical meaning only inside a defined referral pathway with a named clinician responsible for the decision, a defined route for OOD-flagged and ungradable images, and a defined action on a negative result. No such pathway is specified, validated, or supplied. At the internal operating point the module produced 516 false positives against 240 true positives on 8,837 images; a pathway that cannot absorb that referral volume would be harmed, not helped, by this module.

4. **Risk of use outside the intended input type — demonstrated, not hypothetical.** On optic-disc crops this module is **below chance** (AUC 0.3667; 95% CI 0.3145-0.4164), meaning its outputs are systematically anti-correlated with the truth on that input type, while its calibration simultaneously collapses. A user who feeds disc crops to this model receives confident, wrong, badly calibrated answers. The advisory OOD gate has **never been tested on those inputs** and cannot be relied on to prevent this.

5. **Preprocessing mismatch is a silent failure channel.** The same weights produce AUC 0.8729 or 0.9988 on ORIGA depending only on whether the training-domain preprocessing is applied, and REFUGE-Val moves in the opposite direction. A deployment that applies the wrong preprocessing will not fail loudly; it will produce plausible-looking probabilities of unknown quality.

6. **Population gaps as an equity issue.** African populations are essentially absent from training and evaluation, and the G1020 dataset's own provenance is recorded inconsistently across artifacts. The screening gap that motivates this work concentrates in exactly the settings least represented in the data.

7. **Reference-standard conflation.** Near-ceiling AUCs measured against cup-disc-ratio-derived or coarse diagnosis-category labels are not equivalent evidence of glaucoma discrimination in a screening population. Presenting them as such would overstate what has been shown.

---

## 12. Limitations and failure modes

Stated bluntly, and drawn from the manuscript's Limitations section and the glaucoma verification manifest.

1. **Below-chance performance on optic-disc crops.** RIM-ONE DL: **AUC 0.3667 (95% CI 0.3145-0.4164)** under minimal preprocessing, and 0.4735 (0.4209-0.5264) under training-domain preprocessing. Both intervals sit at or below the chance line. Sensitivity at the fixed threshold was 0.1802 (minimal). This is a hard intended-use boundary: the module is a full-fundus-photograph classifier and must not be applied to optic-disc crops.

2. **Degraded performance on cropped fields of view.** ACRIMA: **AUC 0.7185 (0.6824-0.7529)** under minimal preprocessing and **0.6049 (0.5648-0.6465)** under training-domain preprocessing — the latter is worse. Sensitivity 0.3611 / 0.5202. Calibration collapses in the same direction: Brier 0.511797, ECE 0.527851 under the minimal protocol.

3. **Near-chance result on the G1020 dataset-family holdout.** AUC **0.5896 (0.4792-0.6975)** in the evaluation pipeline and **0.5813 (0.4726-0.6846)** in an independent Ben Graham run. Sensitivity 0.4138 and 0.3793. Both near chance with wide CIs, on a small (n=155), label-noisy (initial data-quality score 0.6843657817109144; 23.5% flagged), distribution-shifted (28.39% OOD flag rate) split. The cause is unresolved.

4. **The ORIGA 0.9988 Ben Graham result exceeds all published figures and is unconfirmed.** The trained-on-ORIGA literature ceiling is 0.831-0.851 and well-executed zero-shot models report 0.85-0.854. This result is reported as stored and treated as **unconfirmed pending independent replication**. Neither an undetected evaluation artifact nor undetected training exposure via the unrecorded auxiliary composition can be excluded.

5. **DRISHTI-GS and FIVES are dataset-family holdouts, not strictly zero-shot.** Their train splits contributed auxiliary training data per the module report. Both reach a degenerate AUC 1.0000 under training-domain preprocessing with boundary-degenerate CIs [1.0000, 1.0000]. **These are not external validation and must never be cited as such.** A family-holdout dataset reaching exactly 1.0 is also consistent with the model recognizing sibling-split dataset statistics rather than with genuine generalization. DRISHTI-GS additionally has n=51 with only 13 negatives and 74.5% prevalence.

6. **Threshold prevalence sensitivity.** The fixed threshold 0.044776 was selected on a 2.92%-prevalence partition by the rule maximize-specificity-subject-to-sensitivity >= 0.90 (recovered from released source, not from run artifacts; paper Section 2.4). Its bootstrap specificity CI is wide (**0.7172-0.9724**). At different deployment prevalences the achieved sensitivity/specificity trade-off will shift. All PPV and NPV values in this card are prevalence-dependent and were measured on cohorts whose prevalence ranges from 2.92% to 74.51%.

7. **The training corpus is recoverable only from captured standard output, and no overlap check has been run — both condition every zero-shot label.** The nine-dataset auxiliary composition is now itemized from the pipeline execution log (Section 3), superseding the module report's list, which **omits G1020 despite its confirmed use**. But the training code never persists a dataset manifest, so a rerun on a differently provisioned machine would silently pool a different corpus, and **no image-hash overlap check between the training pool and any external benchmark has been run.** The zero-shot labels for REFUGE-Val and ORIGA, including the ORIGA result in item 4, are conditional on that pending check.

8. **The OOD gate is untested on the declared out-of-intended-use inputs.** No OOD flag rates exist for RIM-ONE DL or ACRIMA. Whether the gate catches the failure modes in items 1 and 2 is unknown.

9. **Calibration drift under shift.** Held-out ECE is 0.003616 in-domain, but mean external ECE under the minimal protocol is **0.219498** across the four full-fundus benchmarks, reaching 0.401042 on DRISHTI-GS-Test and 0.527851 on ACRIMA. Venn-Abers guarantees hold under i.i.d. sampling and do not transfer to shifted prevalence. In-sample Venn-Abers ECE (0.001493) is a fitting diagnostic, not a performance claim.

10. **The interval-to-scalar rule is absent from the run artifacts.** The rule is now specified from the released source (paper Section 2.4): the Vovk regularized point estimate p = p1 / (1 − p0 + p1), denominator floored at 1e-15, clipped to [0, 1]. The residual gap is that the deployed decision rule is evidenced by source code rather than by any artifact of the executed run.

11. **Photograph-based glaucoma detection has an intrinsic ceiling.** Glaucoma cannot be confirmed or excluded from a single fundus photograph. A negative output does not exclude glaucoma.

12. **Reference-standard heterogeneity across benchmarks.** ORIGA labels are substantially cup-disc-ratio-derived; FIVES "glaucoma" is a coarse diagnosis category whose negatives include AMD and DR eyes; DRISHTI-GS is a segmentation dataset with 74.5% prevalence. Cross-dataset results conflate structural suspicion with clinical diagnosis to differing degrees.

13. **The module report contradicts itself on the parameter count** (53.2M in one section, 54.1M in another). The pipeline log records 53,186,549, all trainable (paper Section 2.3), and that value governs Section 1 of this card; the report-level contradiction is disclosed rather than silently resolved.

14. **Split-integrity claims are partially unverified.** The module report **both** asserts an AIROGS patient-level split **and** states that public AIROGS lacks patient identifiers. No patient-split verification artifact exists. If images from the same patient straddle train and evaluation partitions, the internal AUC (0.9792) and the calibration estimates are optimistically biased.

15. **Per-dataset CIs use 1,000 resamples, below the project's own convention.** The bootstrap protocol is recovered from the released source (percentile type, unstratified, fixed seeds; paper Section 2.7): per-dataset AUC/sensitivity/specificity CIs use 1,000 resamples (seeds 42/43/44) while operating-point CIs use 2,000. The 1,000-resample count falls short of the project's 2,000-resample reporting standard, and the counts are recorded in source rather than in the evaluation artifacts.

16. **Preprocessing-effect deltas have no CIs.** The per-dataset deltas in Section 7.4 are point differences. No paired bootstrap CIs were computed, so none of the deltas — including the two negative ones — is shown to be statistically distinguishable from zero.

17. **Training configuration is evidenced by source, not by run artifacts; GPU hardware is unrecorded anywhere.** Optimizer, learning rates (resolving the module reports' 3e-4-vs-1e-4 contradiction), weight decay, scheduler, loss, batch size, precision, and seed were recovered from the released source code and are asserted in Section 1 on that basis (paper Table 2). The GPU hardware of the executed run remains unrecorded in both artifacts and source.

18. **G1020 provenance conflict.** One artifact describes G1020 as a "Pakistani population" cohort; the manuscript's dataset table records a German private clinical practice in Kaiserslautern with 432 patients. This conflict is unresolved and affects any population-level interpretation of the G1020 result.

19. **Single training run, single seed.** Run-to-run variance is unquantified; no repeat-run artifact exists.

20. **Per-image, not per-patient.** All operating characteristics are per-image. Clinical screening decisions are per-patient. No per-patient aggregation rule is defined or validated.

21. **No gradability or imageability pathway.** All benchmarks are curated and gradability-filtered. Prospective imageability is unmeasured. No mydriasis, field, or resolution input specification is defined.

22. **Retrospective only.** Every result is retrospective and computed on curated public datasets. No prospective, intent-to-screen, or real-device validation exists. The generalization gap documented for this field — internal AUCs of 0.986-0.996 decaying to 0.923 on multiethnic external data and 0.823 on website-sourced images, and a single fixed model spanning AUC 0.769-0.987 across 13 external sources — should be assumed to apply here.

23. **No regulatory status.** RETGUARD has no regulatory clearance, authorization, or certification anywhere. It is not a medical device.
