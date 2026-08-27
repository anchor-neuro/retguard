# Model Card — RETGUARD AMD/DME Module (OCT)

**Model card version:** 1.1 | **Date:** 2026-08-27 | **Status:** Research software. No regulatory clearance.

> **This is research software. It is not a medical device. It has no FDA clearance, no CE mark, and no regulatory authorization in any jurisdiction. It must not be used to make or inform any clinical decision about any patient.**
>
> **The "Normal" output of this module means "no AMD or DME detected." It does not mean "healthy." In the confounder stress test, 347 of 354 eyes with genuine non-target pathology — including retinal artery occlusion, an ocular emergency — were output as "Normal." See Sections 7.5, 11, and 12.**

This card follows the Model Cards for Model Reporting framework (Mitchell M, Wu S, Zaldivar A, et al. Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*. 2019:220-229. DOI: 10.1145/3287560.3287596).

Every number in this card is copied from the RETGUARD manuscript evidence base or from the internal structured-verification manifest `manifest_AMD-DME.json` (generated 2026-08-22). Nothing is estimated, and nothing is rounded beyond the precision the source artifact stores. Values that verification could not trace to an artifact are labeled **report-stated** wherever they appear.

---

## 1. Model details

| Item | Value |
|---|---|
| Developer | Sameh Aboelmaaty, Anchor Neuro, Delaware, USA |
| Model card date | 2026-08-27 |
| Module version | RETGUARD OCT module, **RUN-011** (three-class). RUN-010 was a preceding binary AMD-vs-Normal model; **no RUN-010 number describes this model** |
| Model type | Three-class image classifier: Normal (0) / AMD (1) / DME (2) |
| Architecture | EfficientNetV2-M, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k (report-stated); export metadata records only "efficientnetv2_m + full fine-tune (eval mode)" |
| Input | Single OCT **B-scan**, 480 x 480 x 3 |
| Trainable parameters | **53,187,063 — report-stated.** No artifact records a parameter count for this module |
| Outputs (ONNX) | `probability` (batch, 3) softmax over [Normal, AMD, DME]; `logit` (batch, 3) raw logits; `p_amd` (batch,) convenience output = probability[:, 1] |
| Decision rule (deployed) | P(AMD) >= 0.70 → AMD positive. An argmax rule is also used in some artifacts; the two are labeled separately throughout this card |
| Calibration | Inductive Venn-Abers Predictor (IVAP) on P(AMD) in packaged inference; a fitted temperature (0.819011) is retained as an evaluation artifact but is not a runtime fallback. Initialization fails closed without a valid Venn-Abers artifact |
| Test-time augmentation | 2 views: identity and horizontal flip (verified from released source; paper Table 2) |
| Out-of-distribution gate | Mahalanobis-distance detector on penultimate features, 97th-percentile threshold 31.2676; advisory only |
| Export format | ONNX, opset 17, input size 480; audited as `retguard_amd_effnetv2m_480px.onnx` (no file size recorded in the audit artifacts); released as `retguard_oct_v1.0.0.onnx` + `retguard_oct_v1.0.0.onnx.data` — the same export, renamed, with CC BY-NC 4.0 metadata embedded (Section 10). TensorRT precision field FP16 |
| Export verification | **Re-verification pending.** The report claims all parity checks passed (maximum probability difference 0.000287; maximum logit difference 0.000987) but **no artifact evidences these values** |
| Training seed | 42, all stages (verified from released source; single run) |
| License — weights | CC BY-NC 4.0, research use; commercial licensing available separately (see LICENSE-WEIGHTS.md, COMMERCIAL-LICENSE.md) |
| License — code | PolyForm Noncommercial 1.0.0 (see LICENSE.md) |
| Contact | ghoneim2012@gmail.com |
| Repository | https://github.com/anchor-neuro/retguard |
| Public weights | GitHub release `v1.0.0`; no public Hugging Face mirror is currently verified |

**Paper citation.** Aboelmaaty S. RETGUARD: Retrospective Multi-Dataset Development and External Testing of Calibrated Deep-Learning Classifiers for Retinal Fundus Photographs and OCT B-Scans. Version 1 submitted to medRxiv; public DOI pending screening, 2026-08-27. Anchor Neuro, Delaware, USA.

**Input preprocessing (the shipped recipe, in this exact order; parity-verified against the released source):** (1) border crop by grayscale threshold 10 with a bounding rectangle, applied only if the region of interest exceeds 30% of both original dimensions; (2) resize to 480 x 480 with Lanczos4 interpolation; (3) grayscale conversion; (4) percentile normalization — clip to the 1st and 99th percentiles and rescale to [0, 255], skipped if that range is degenerate; (5) CLAHE, clip limit 2.0, 8x8 tile grid; (6) replication of the grayscale channel to three channels; then, at model input, ImageNet normalization (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).

**Class mapping.** Kermany source classes were mapped to three labels: choroidal neovascularization (CNV) **and** drusen both to **AMD**; NORMAL to Normal; DME to DME. **This mapping collapses neovascular and non-neovascular AMD into one output and therefore cannot assign referral urgency** (Section 12, item 5).

**Verified training configuration:** 19 epochs run; best epoch 7 (validation composite 0.9904, the maximum in the history; validation AUC 0.9987, sensitivity 0.9925, specificity 0.9885, F1 0.9922, Brier 0.021805, MCC 0.9764, ECE 0.01442, threshold 0.5407, backbone LR 5.41e-05, head LR 5.96e-04). Overfitting evidence is visible in the history: train loss 0.0706 at epoch 7 falls to 0.0582 at epoch 19 while validation loss rises from 0.0202 to 0.0300. Validation ECE fell from 0.043455 (epoch 1) to 0.006193 (epoch 19), **but not monotonically** — it rose at epochs 5→6, 10→11, 11→12, 14→15, and 17→18.

**Training-configuration items that are report-stated only, with no artifact backing, and are therefore not asserted as values here:** total training-sample count and per-domain composition, domain balancing percentage, effective samples per epoch, batch size and gradient accumulation, optimizer, base learning rates, weight decay, layer-wise learning-rate decay, EMA decay, focal gamma, MixUp alpha, early-stopping patience, class weights, spatial-attention loss parameters, GPU hardware, and PyTorch version.

---

## 2. Intended use

**Primary intended use.** Research and methodological evaluation only: retrospective benchmarking of three-class Normal/AMD/DME classification from OCT B-scans, calibration behavior, out-of-distribution flagging, and confounder robustness on public OCT datasets.

**Primary intended users.** Machine-learning researchers and clinical-AI evaluators working with de-identified, publicly available OCT datasets.

**Required input type.** A single macular **OCT B-scan**, preprocessed as specified in Section 1, resized to 480 x 480.

**Out of scope — this module must not be used for:**

- Any clinical decision, triage, referral, diagnosis, or exclusion of disease for any patient.
- Any use implying regulatory clearance. There is none, in any jurisdiction.
- **Ruling out disease.** A "Normal" output means only "no AMD or DME detected by this module." It is not a statement that the macula is healthy. There is no "other pathology" or "cannot classify" output class.
- **Assigning referral urgency for AMD.** CNV and drusen are collapsed into a single AMD class, so the module cannot distinguish neovascular AMD (urgent anti-VEGF referral) from early or intermediate AMD (routine monitoring).
- Any imaging type other than an OCT B-scan — not fundus photographs, not OCT angiography, not en-face images.
- **Volume-level or per-patient decisions.** No valid volume-level evidence exists (Section 7.6).
- Any autonomous or unattended operation.
- Deployment on any OCT device or in any population without prospective, target-device validation and local recalibration.

---

## 3. Factors

**Populations and devices represented in training:**

| Dataset | Origin / device | Role |
|---|---|---|
| Kermany ("OCT2017") | Spectralis; multi-center US and Chinese cohorts (per the dataset publication) | Train + internal evaluation (47,068 post-cleaning) |
| Noor Eye Hospital | Tehran, Iran; Heidelberg SD-OCT | Auxiliary train (6,932 — report-stated) + evaluation of the full set |
| OCTDL | Optovue Avanti RTVue XR | Auxiliary train (1,444 — report-stated) + evaluation of the full set + 354-image confounder pool |

**Devices represented in strictly external, never-trained-on evaluation:** Duke (Srinivasan) — Spectralis SD-OCT under a different acquisition protocol; OCTID — Cirrus HD-OCT (Zeiss).

Device manufacturer attributions are recorded in the module report and the manuscript's dataset table. The executed-run artifacts do not back device strings, dataset terms, or source repositories; those fields were rechecked separately against identified source pages on 2026-08-27 and remain subject to the limitations below.

**Known geographic and ethnic gaps.** Training and evaluation data derive from US, Chinese, Iranian, and Russian-hosted (OCTDL) cohorts plus two US test-only sets. **African populations are essentially absent from both training and evaluation.** Performance in those populations is unknown, not merely unproven. No per-subgroup analysis by age, sex, ethnicity, or scan quality was performed.

**Disease-subtype factor.** The Duke and Noor cohorts are described as dry-AMD, but Duke performance is withdrawn and Noor includes training images. The remaining reportable external OCTID result is small and contains no DME. Zero-shot CNV detection is therefore largely unmeasured.

**DME coverage factor.** **Noor and OCTID contain zero DME cases.** The archived Duke DME result is withdrawn after the released-model reproduction gate failed, so no validated strictly external DME performance estimate remains.

**Label-quality factor.** In confident-learning cleaning of the Kermany training data, the DME class had the lowest per-class data-quality score (0.5938) and the lowest probe recall (0.6296), which fell below the 0.65 probe-reliability threshold. **A reliability gate consequently prevented any DME-class label edits** (0 DME removals), meaning the DME class received the least label-cleaning intervention and now has no validated strictly external performance estimate.

---

## 4. Metrics

| Metric | Why it is reported |
|---|---|
| AMD-vs-rest AUC with 95% bootstrap CI | Primary discrimination metric; threshold-free and portable across datasets |
| Macro three-class AUC | Overall three-class discrimination. **Never valid for OCTID or Noor**, which contain zero DME cases — the stored OCTID macro AUC of 0.5 is degenerate |
| Per-class sensitivity and specificity (Normal, AMD, DME) | Directional class errors are important, while DME is invisible in an AMD-vs-rest summary and lacks a validated strictly external estimate |
| Sensitivity and specificity at the released threshold rule P(AMD) >= 0.70, with bootstrap CIs | The released research-interface operating characteristic; not a clinically derived threshold |
| Accuracy, MCC, full confusion matrix | Three-class performance including the direction of each error, which determines whether a miss still generates a referral |
| ECE, MCE, Brier score | Experimental calibration diagnostics, reported separately for **in-sample** assessment (the calibrator's own fitting partition, optimistically biased by construction) and **held-out** assessment |
| Venn-Abers interval width | Per-input uncertainty signal from the distribution-free calibrator |
| OOD flag rate per dataset | Measures how often the advisory gate flags a scan; no validated clinical review pathway exists. Reported as a diagnostic, never tuned |
| Confounder false-AMD rate, and the full prediction breakdown | Measures both a clinically relevant false-positive mode and — with equal prominence — the false-reassurance mode that standard benchmarks do not probe |

**Statistical protocol.** All retained CIs are bootstrap-based, of the percentile type, unstratified, with fixed seeds (verified from released source; paper Section 2.7). The operating-point threshold, sensitivity, and specificity CIs use 2,000 resamples (seed 42); most per-dataset AUC, sensitivity, and specificity CIs use 1,000 resamples (seeds 42, 43, and 44 respectively) — below the project's own 2,000-resample reporting convention. ECE was computed with 15 bins in the temperature-scaling block. Slightly different OCTID AUC bounds exist in two artifacts; both are image-level. Every archived Duke interval and operating characteristic is withdrawn because the released-model reproduction gate failed.

---

## 5. Training data

**Reported composition:** 55,444 training images — Kermany 47,068 (**verified**) plus Noor 6,932 (**report-stated**) plus OCTDL 1,444 (**report-stated**) — domain-balanced at 33.33% each (report-stated). The 55,444 total and the two auxiliary counts have no artifact backing.

**Kermany label cleaning (confident learning, verified):**

| Item | Value |
|---|---|
| Initial n | 49,889 |
| Final n | **47,068** |
| Issues flagged | 4,877 (9.78%) |
| Label corrections | 159 (60 flips 0→1; 99 flips 1→0) |
| Removals | 455 |
| Excluded as uncertain | 2,366 |
| Skipped for insufficient confidence | 1,897 |
| Data-quality score before | 0.6500831846699673 |
| Data-quality score after (all samples) | 0.6532702599771493 |
| Data-quality score after (kept samples only) | 0.6913811848472574 |
| Initial class distribution | Normal 16,419 / AMD 26,782 / DME 6,688 |
| Final class distribution | Normal 16,007 / AMD 25,362 / DME 5,699 |
| Issues per class | Normal 786 / AMD 3,102 / DME 989 |
| Removals per class | Normal 79 / AMD 376 / **DME 0** |
| Per-class data-quality score | Normal 0.7495 / AMD 0.6768 / **DME 0.5938** |
| Per-class probe recall | Normal 0.8041 / AMD 0.7574 / **DME 0.6296** |
| Probe judged reliable | Normal yes / AMD yes / **DME no** |
| Probe-reliability threshold | 0.65 |
| Protections | 5 CV folds; within-boundary confidence 0.92; cross-boundary confidence 0.95; fold-agreement threshold 0.60; 5% maximum-removal safety stop; probe MLP(1280→512→3) + BN + ReLU + Dropout(0.3) |
| Cleaning wall time | 42.56861925125122 seconds |

**The DME class received zero label edits** because its probe recall (0.6296) fell below the 0.65 reliability threshold. This is a deliberate safety gate, and it means the class with the weakest label quality is also the class whose labels were least corrected.

**Reserved confounder pool.** OCTDL non-AMD/non-DME pathologies — epiretinal membrane (ERM), retinal artery occlusion (RAO), retinal vein occlusion (RVO), and vitreomacular interface disease (VID) — were reserved as a 354-image pool and **never trained on as AMD or DME**.

**License and access status of the datasets used** (rechecked against identified source pages on 2026-08-27; these terms are not encoded in the executed-run artifacts and do not by themselves establish learned-weight redistribution rights):

| Dataset | License / access terms |
|---|---|
| Kermany ("OCT2017") | CC BY 4.0 (Mendeley Data) |
| Noor Eye Hospital | CC BY 4.0 (Mendeley Data v1) |
| OCTDL | CC BY 4.0 (Zenodo) |
| Duke (Srinivasan) | Research/educational use; **no redistribution** |
| OCTID | CC0 1.0 on the Borealis component records |

These summaries were rechecked against the identified source pages on 2026-08-27. **Learned-weight redistribution and commercial use are not established as permitted for all contributing datasets and require rights-holder review; Duke expressly prohibits commercialization and raw-data redistribution.**

---

## 6. Evaluation data

Classification uses exactly the manuscript's terms.

| Evaluation set | n evaluated | Device | Classification (manuscript's exact term) |
|---|---|---|---|
| Kermany-Test | 11,129 (Normal 3,550 / AMD 5,983 / DME 1,596) | Spectralis-dominant | **Internal held-out** (patient-level split is report-stated; no verification artifact) |
| Kermany threshold/calibration partition | 14,384 | Spectralis-dominant | In-sample for the fitted calibrators |
| **OCTDL** | **2,064** (Normal 686 / AMD 1,231 / DME 147) | Optovue Avanti RTVue XR | **Includes training portion — NOT external.** 1,444 of these 2,064 images were used in training (report-stated) |
| **Noor** | **9,904** (Normal 1,607 / AMD 8,297 / **DME 0**) | Heidelberg SD-OCT | **Includes training portion — NOT external.** 6,932 of these 9,904 images were used in training (report-stated) |
| **Duke (Srinivasan)** | **3,231 scans in 45 volumes** (15 Normal / 15 AMD / 15 DME volumes) | Spectralis SD-OCT, different protocol | **External cohort; performance claims withdrawn after failed released-model reproduction gate** |
| **OCTID** | **261** (Normal 206 / AMD 55 / **DME 0**) | Cirrus HD-OCT (Zeiss) | **External (zero-shot; test-only); small sample** |
| OCTDL confounder pool | 354 (ERM, RAO, RVO, VID) | Optovue Avanti RTVue XR | Stress test; never trained as AMD or DME |

**The cohorts designated strictly external and never trained on are Duke and OCTID.** The released-model Duke performance claim is withdrawn after a failed reproduction gate; only OCTID retains a reportable external performance result. The OCTDL and Noor evaluations were run on their **full** image sets, which include the images used in training. Those two rows measure in-domain performance, not external validation. The internal evaluation artifact itself lists only Kermany-Test as release-gate eligible.

**Because Noor and OCTID contain zero DME cases and the archived Duke claims are withdrawn, no validated strictly external DME performance estimate remains.**

**Unexplained attrition.** The module report states source counts of 16,822 for Noor and 572 for OCTID; the evaluated counts are 9,904 and 261. **The attrition rule is not documented.** All performance denominators in this card are the artifact-recorded evaluated counts. OCTID's excluded classes (macular hole, central serous retinopathy, diabetic retinopathy) are pathologies of the same kind the confounder stress test probes; their exclusion is disclosed as a gap, not as a design choice with a recorded rationale.

**Partition-accounting gap.** The Kermany validation-split count (8,501) is report-stated. The accounting from the source distribution to the four partitions (49,889 + 8,501 + 14,384 + 11,129) is not fully artifact-documented. Kermany-Test is a **custom re-split**, not the canonical 1,000-image Kermany test set used by much of the published literature.

---

## 7. Quantitative results

### 7.1 Internal held-out evaluation (Kermany-Test, n = 11,129)

| Quantity | Value |
|---|---|
| AMD-vs-rest AUC | **0.9990075521010786** — **no stored 95% CI anywhere in the artifacts** |
| Macro three-class AUC | 0.9987420856215916 |
| Accuracy | 0.9805013927576601 |
| MCC | 0.9670126295890691 |
| Brier | 0.030522906651766 |
| ECE (held-out) | 0.019077097724760676 |
| TTA | 2 views |
| OOD flagged | 520 (4.67%); score mean 17.6721, maximum 91.1425 |

**Per-class performance:**

| Class | Sensitivity | Specificity | PPV | NPV | F1 | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| Normal | 0.9890140845070422 | 0.9866737036548358 | 0.9720376522702104 | 0.9948117600106425 | 0.9804523876012287 | 3,511 / 101 / 39 / 7,478 |
| AMD | 0.981280294166806 | 0.9912553439564711 | 0.992393509127789 | 0.9785152503356992 | 0.986805613917136 | 5,871 / 45 / 112 / 5,101 |
| DME | 0.9586466165413534 | 0.9925521871394105 | 0.9556527170518426 | 0.993073047858942 | 0.9571473256177666 | 1,530 / 71 / 66 / 9,462 |

**Confusion matrix** (rows = true, columns = predicted, order Normal / AMD / DME):

|  | pred Normal | pred AMD | pred DME |
|---|---|---|---|
| **true Normal** | 3,511 | 20 | 19 |
| **true AMD** | 60 | 5,871 | 52 |
| **true DME** | 41 | 25 | 1,530 |

**Direction of the errors matters.** Of 112 AMD images missed by the three-class argmax rule (1.87% of 5,983), 52 had DME as the highest softmax class and **60 had Normal as the highest class**. Of 66 missed DME images, **41 were classified as Normal** and 25 as AMD. The released Boolean `decision` is separate from argmax: it is an AMD-only rule based on Venn-Abers-calibrated P(AMD) >= 0.70. The released interface has no validated DME threshold and no DME-inclusive referral field, so a DME argmax must not be described as necessarily generating a referral.

### 7.2 Cross-dataset evaluation, AMD vs rest

Sensitivity and specificity in this table use the **deployed threshold rule, P(AMD) >= 0.70**.

| Dataset (device) | n | Exposure | AUC (95% CI) | AMD sens. | AMD spec. | OOD flagged |
|---|---|---|---|---|---|---|
| Kermany-Test (Spectralis-dominant) | 11,129 | internal holdout | 0.9990 (**no stored CI**) | 98.13% | 99.13% | 4.7% |
| OCTDL (Optovue Avanti RTVue XR) | 2,064 | **includes training portion — not external** | 0.9990 (0.9980-0.9998) | 99.27% | 99.76% | 38.4% |
| Noor (Heidelberg SD-OCT; no DME cases) | 9,904 | **includes training portion — not external** | 0.9999 (0.9998-0.9999) | 99.55% | 99.07% | 66.6% |
| **Duke Srinivasan (Spectralis, different protocol)** | **3,231 scans / 45 volumes** | **external cohort** | **Not reported - reproduction gate failed** | **Not reported** | **Not reported** | **Not reported** |
| **OCTID (Cirrus HD-OCT; 55 AMD positives, no DME cases)** | **261** | **external (zero-shot; test-only); small sample** | **0.9999 (0.9995-1.0000)** | **98.18%** | **100.0%** | 13.8% |

Alternative image-level OCTID bounds from `benchmark_results.json` are [0.9997451213103052, 1.0] for point AUC 0.9999117387466903. Historical Duke point estimates and intervals are withdrawn and must not be cited.

Retained stored confusion matrices (AMD vs rest): OCTDL TP 1,222 / TN 831 / FP 2 / FN 9; Noor TP 8,260 / TN 1,592 / FP 15 / FN 37; **OCTID TP 54 / TN 206 / FP 0 / FN 1**. The OCTID row rests on 55 AMD positives, and a single false negative produces its 98.18% sensitivity. The archived Duke confusion matrices are withdrawn.

The archived four-dataset mean AUC/sensitivity/specificity summary is withdrawn because it includes the non-reproduced Duke result and the in-domain OCTDL/Noor evaluations; it does not describe external performance.

**Never cite macro AUC for OCTID** (stored value 0.5, degenerate — zero DME samples) **or any DME metric for OCTID or Noor** (both have zero DME cases).

### 7.3 Duke reproduction failure and claim withdrawal

The official Duke distribution contains 3,231 B-scans nested within 45 source volumes (15 Normal, 15 AMD, and 15 DME volumes). The released v1.0.0 model processed every scan with zero exclusions, but it failed the prespecified requirement to reproduce the archived scan-level AMD-vs-rest AUC. Six bounded checks covering the documented direct/JPEG95 preprocessing and identity/probability-mean/logit-mean score variants also failed, while the archived and released ONNX graphs were proven functionally identical. Accordingly, **all archived Duke discrimination, threshold, confusion-matrix, DME, calibration, OOD, and volume-level values are withdrawn**. No replacement point estimate, confidence interval, sensitivity/specificity, or cluster/volume statistic was generated.

### 7.4 OCTID (external, small sample)

Accuracy 0.9961685823754789; MCC 0.9884710915419982; Brier 0.009198395123455683; ECE 0.021204965024958167; AMD sensitivity 0.9818181818181818, specificity 1.0, PPV 1.0, NPV 0.9951690821256038, F1 0.9908256880733946; confusion TP 54 / TN 206 / FP 0 / FN 1. Zero DME cases. **Stored macro AUC is 0.5 and is degenerate — never cite it.**

### 7.5 Confounder stress test (354 OCTDL images of pathologies never trained as AMD or DME)

| Predicted class | Count | Share |
|---|---|---|
| **Normal** | **347** | **98.0%** |
| AMD | 2 | 0.56% |
| DME | 5 | 1.41% |

False-AMD rate: **0.56%** under the argmax rule (`fpr_argmax` = 0.0056, matching the 2/354 breakdown) and **0.28%** under the deployed threshold rule (`fpr_threshold` = 0.0028).

**Both halves of this result must be read together.** The low false-AMD rate measures a clinically relevant failure — falsely reporting AMD in the presence of other macular disease — that standard benchmarks do not probe. Its flip side is a serious hazard of equal prominence: **347 of 354 eyes with genuine pathology, including retinal artery occlusion (an ocular emergency) and retinal vein occlusion, received the output "Normal."** The three-class head has no "other pathology / cannot classify" output. **The OOD flag rate within these 354 confounder images specifically is not recorded in the artifacts**, so whether the gate would route these eyes to human review is unknown.

The module report additionally states a 14.1% false-AMD rate for the preceding binary RUN-010 design. That figure is **report-stated** (not traceable to RUN-010 artifacts), and the before/after comparison is uncontrolled across runs (different task, data, and configuration), so **no causal attribution or reduction multiplier is claimed**.

### 7.6 Volume-level evaluation — withdrawn

The Srinivasan cohort contains 45 source volumes (15 Normal, 15 AMD, and 15 DME), but the archived pipeline collapsed folder-derived identities into class labels. Its stored aggregation therefore did not represent the 45 independent volumes. The historical volume-level metrics and gate are withdrawn and are not treated as evidence. No corrected volume-level estimate is reported because the released-model scan-level reproduction gate failed before downstream inferential analysis was authorized.

### 7.7 Explainability and release gate

The archived explanation artifact records 50 AMD-positive test images, all with decision "AMD" and calibrated probabilities in the range 0.9892-0.9996. The module adds a spatial-attention regularizer during training that penalizes activation outside the outer-retina zone (rows 6-8 inclusive of the 15 x 15 feature map; the module report's "rows 6-9" quotes the exclusive slice bound — paper Section 2.4). **The reported Grad-CAM zone statistics — mean 38.99% activation within the AMD zone, 6 of 50 images below a 30% threshold, and a 25% RUN-010 baseline — are report-stated and are stored in no readable artifact.** They must be recomputed from the archived per-image files before being treated as verified. Neovascular disease may extend anterior to the regularized band, and consistent vertical centering of B-scans is not evidenced.

The automated release gate passed (`kill_gate_passed = true`, zero failures recorded). The achieved values (AUC 0.9990; AMD sensitivity 0.9813; confounder false-AMD rate 0.56%) are artifact-verified, but **the numeric gate thresholds quoted in the run report (Kermany-Test AMD AUC >= 0.945; AMD sensitivity >= 0.85; confounder false-positive rate <= 20%) are report-asserted — the threshold values are not stored in the evaluation artifacts.** The additional historical Duke volume gate is withdrawn because it relied on the defective aggregation described in Section 7.6.

---

## 8. Calibration

| Item | Value |
|---|---|
| Calibrator types | Temperature scaling, plus an Inductive Venn-Abers Predictor (IVAP, Vovk & Petej 2014) |
| Temperature | **0.819011** (the manuscript rounds this to 0.819) |
| Fitting partition | **14,384 Kermany images** (the threshold/calibration partition) |
| Packaged calibration behavior | `venn_abers` required on P(AMD); initialization fails closed if the artifact is missing, malformed, or incompatible. `temperature_scaling` is an evaluation artifact, not a runtime fallback |
| ECE bins | 15 (temperature-scaling block) |

**In-sample assessment** — computed on the same 14,384-image partition used to fit the calibrators, and therefore **optimistically biased by construction**:

| Calibrator | Brier | ECE | MCE |
|---|---|---|---|
| Temperature scaling | 0.006529 | 0.008629 | 0.333447 |
| Venn-Abers (IVAP) | 0.00534 | **0.000271** | 0.735843 |

**For isotonic-family calibrators such as IVAP, near-zero in-sample ECE is close to guaranteed by construction.** The 0.000271 figure is a fitting diagnostic, not evidence of deployed calibration quality, and must never be quoted as a deployment calibration result.

Venn-Abers interval widths on the fitting partition: mean 0.000139, median 0.0, **maximum 0.224764**, SD 0.00281.

**Held-out assessment** — the honest post-calibration estimate:

| Partition | ECE | Brier |
|---|---|---|
| **Kermany-Test (held-out)** | **0.019077097724760676** | 0.030522906651766 |

**Calibration under distribution shift remains uncertain.** On OCTID, the stored image-level ECE is 0.021204965024958167, compared with held-out in-domain ECE 0.019077097724760676. The archived Duke calibration and defective three-unit volume values are withdrawn with the non-reproduced performance claim.

**Operating point at P(AMD) >= 0.70**, measured on the threshold partition (n = 14,384): sensitivity 0.9917 (95% CI 0.9897-0.9936), specificity 0.9940 (95% CI 0.9919-0.9959), AUC 0.999286 (95% CI 0.9989-0.9996), 2,000 bootstrap iterations. **This AUC and CI belong to the threshold partition, not the held-out test set.**

**The 0.70 threshold has no documented clinical derivation.** It is recorded in the operating-point artifact as a pre-specified threshold cap (`max_clinical_threshold` = 0.7, `ceiling_applied` = true). Because 0.70 was imposed as a cap, **every bootstrap resample returned the same value**, so the threshold CI is a single point (mean 0.7, SD 0.0). The underlying selection rule — maximize specificity subject to sensitivity >= 0.90 — is not stored in the artifacts; it was recovered from the released source code (paper Section 2.4), and in the executed run the cap bound instead of the rule-selected value.

**Interval-to-scalar rule.** The Venn-Abers interval is reduced to the scalar probability the fixed threshold consumes by the Vovk regularized point estimate, p = p1 / (1 − p0 + p1), with the denominator floored at 1e-15 and the result clipped to [0, 1] — not the interval midpoint and not p1 alone (not recorded in the evaluation artifacts; verified from released source, paper Section 2.4, and implemented identically in the accompanying inference package). Venn-Abers validity holds under i.i.d. sampling and does not transfer to shifted prevalence.

---

## 9. Out-of-distribution gate

| Item | Value |
|---|---|
| Method | Mahalanobis distance on penultimate backbone features (method string recorded in the report, not in the JSON artifacts) |
| Threshold percentile | 97th percentile of training-set scores |
| Threshold value | 31.2676 |
| Behavior | **Advisory only.** Flags scans for human review; artifact evaluation counts confirm no samples were excluded from any evaluation |

**Flag rates per dataset:**

| Dataset | Flagged | n | Flag rate | Score mean | Score maximum |
|---|---|---|---|---|---|
| Kermany-Test (same domain) | 520 | 11,129 | 4.67% | 17.6721 | 91.1425 |
| OCTID (Cirrus) | 36 | 261 | 13.79% | 19.3232 | 71.5995 |
| OCTDL (Optovue) | 793 | 2,064 | 38.42% | 32.1886 | 101.9114 |
| Duke (Spectralis, different protocol) | Not reported | 3,231 scans / 45 volumes | Withdrawn after reproduction failure | Not reported | Not reported |
| Noor (Heidelberg, different population) | 6,599 | 9,904 | **66.63%** | 39.6642 | 133.0552 |

**What was and was not tested — stated plainly.**

- **Tested and retained:** flag rates on the four non-Duke evaluation sets above, reported as diagnostics and never tuned. The archived Duke flag rate is withdrawn with the failed reproduction claim.
- **Not tested:** **the OOD flag rate within the 354-image confounder pool is not recorded**, so whether the gate would route eyes with retinal artery occlusion, retinal vein occlusion, epiretinal membrane, or vitreomacular interface disease to human review is **unknown**. This is the single most consequential untested question for this gate, because those are exactly the eyes the classifier outputs as "Normal."
- **Specified from source, not from artifacts:** the detector's exact form is absent from the parseable evaluation artifacts but is now specified from the released source (paper Section 2.4) — a single pooled Gaussian over L2-normalized 1,280-dimensional post-GAP features, empirical covariance plus a 1e-5 ridge, Mahalanobis distance (not its square) via Cholesky forward substitution, fitted on the Kermany training partition alone in this module. The training score range and fitting duration remain inside a binary archive the audit could not parse.
- **Not what it measures:** the gate is not a scan-quality or gradability assessment. It measures feature-space distance only.

**The ordering by nominal domain shift is a post-hoc observation, not a pre-specified prediction, and it is confounded.** Roughly 70% of the evaluated Noor images were in the training set, yet two-thirds of them are flagged — a pattern at least as consistent with a gate sensitive to acquisition characteristics as with clean domain-shift tracking, given that discrimination is preserved on every flagged set. **The gate flags far more scans than the model misclassifies; flag rate did not correlate with observed error.**

**Operational consequence.** A 38-67% flag rate would route one- to two-thirds of scans at a device-shifted site to human review, negating much of the workload rationale for automated screening. Per-site threshold re-tuning would be required for a usable triage rate, and no such re-tuning procedure has been validated.

---

## 10. Deployment artifacts

| Item | Value |
|---|---|
| ONNX file | Audited as `retguard_amd_effnetv2m_480px.onnx`, opset 17, input size 480 (no file size recorded in the audit artifacts); released as `retguard_oct_v1.0.0.onnx` + `retguard_oct_v1.0.0.onnx.data` — the same export, renamed, with CC BY-NC 4.0 metadata embedded and the external-data reference updated to the renamed companion file |
| Outputs | `probability` (batch, 3), `logit` (batch, 3), `p_amd` (batch,) |
| TensorRT precision field | FP16 |
| **Export parity** | **Re-verification pending.** The report claims maximum probability difference 0.000287 and maximum logit difference 0.000987 with all checks passed, but **`export_metadata.json` contains no verification-diff fields and no artifact evidences these values.** OCT export parity is unverified, not verified |
| Companion artifacts required at inference | Venn-Abers calibrator, operating-point configuration, OOD gate archive |
| Released asset (v1.0.0) | `retguard-oct-v1.0.0.zip`: `retguard_oct_v1.0.0.onnx`, `retguard_oct_v1.0.0.onnx.data`, `ood_gate_oct_v1.0.0.npz`, `venn_abers_oct_v1.0.0.npz`, `LICENSE.txt`, this model card |
| SHA-256, `retguard_oct_v1.0.0.onnx` | `5ebd2e814a718edca922c26f5bdce380c6b38506f923103a8cb3362b67fb75f3` |
| SHA-256, `retguard_oct_v1.0.0.onnx.data` | `88d6c4d6803d3201b182eeb528b9ab08f2641de2b1d1ef672c807f9ecda5243c` |
| SHA-256, `ood_gate_oct_v1.0.0.npz` | `68fc53ad187bdb14f2dae53fd41ad25bc563fc7c8d7cad6517d065578fe87532` |
| SHA-256, `venn_abers_oct_v1.0.0.npz` | `176c74b62e8b100d186249dd6f63dac877ca13385c6d38254be774f8cd05a5a4` |
| Integrity | The zip digest and every member digest are published in the v1.0.0 release's `SHA256SUMS.txt` and embedded in `retguard/weights.py`; `retguard verify` re-checks them |

No network connectivity is required at inference time. No ONNX Runtime or edge-hardware benchmark has been run for this module, and no latency measurement of any kind is recorded for it.

---

## 11. Ethical considerations

1. **"Normal" does not mean healthy — the central hazard of this module.** The three-class head can output only Normal, AMD, or DME. There is no "other pathology" and no "cannot classify" class. In the confounder stress test, **347 of 354 eyes with genuine, non-target pathology — including retinal artery occlusion, an ocular emergency, and retinal vein occlusion — were output as "Normal."** In a screening deployment that output risks false reassurance and non-referral of patients who need urgent care. This hazard is stated here with the same prominence as the 0.56% false-AMD rate, because the two are the same measurement seen from opposite sides.

2. **Low-prevalence PPV behavior.** All retained PPV values were measured on disease-enriched research cohorts. The AMD class counts in Section 6 — 5,983 of 11,129 in Kermany-Test, 1,231 of 2,064 in OCTDL, 8,297 of 9,904 in Noor, and 55 of 261 in OCTID — are all far above any plausible screening prevalence. PPV and NPV are prevalence-dependent; no prevalence-standardized projection has been computed.

3. **What a negative result means, and does not mean.** A "Normal" output means only that this module did not detect AMD or DME in this single B-scan. It does not exclude other macular disease, it does not exclude glaucoma or diabetic retinopathy, and it carries no information about pathology outside the imaged region. No validated strictly external DME performance estimate remains after the Duke claim withdrawal.

4. **No referral urgency can be assigned.** CNV and drusen are collapsed into a single AMD class. An "AMD" output cannot distinguish neovascular AMD, which requires urgent anti-VEGF referral, from early or intermediate AMD, which requires routine monitoring. A clinician receiving this output cannot triage on it. External validation additionally covers predominantly dry AMD, so zero-shot CNV detection is largely unmeasured.

5. **Referral-pathway dependence.** The output has clinical meaning only inside a defined referral pathway with a named clinician responsible for the decision, a defined route for OOD-flagged scans, and a defined action on a "Normal" result that accounts for item 1. No such pathway is specified, validated, or supplied. Because the OOD gate flags 38-67% of scans on device-shifted data, a pathway must also be able to absorb that review volume.

6. **Risk of use outside the intended input type.** This is an OCT B-scan classifier. It has never been evaluated on any other modality. The companion glaucoma module, sharing the same architecture and training discipline, performs below chance on an input type it was not built for (AUC 0.3667 on optic-disc crops) — concrete evidence that a RETGUARD classifier can produce confident, systematically wrong output off its intended input. The advisory OOD gate has not been validated as a defense against this.

7. **Per-B-scan, not per-eye.** Retained operating characteristics are per-B-scan. A clinical OCT decision is made on a volume. No valid volume-level analysis exists, so no evidence supports how per-B-scan outputs should be aggregated to an eye-level or patient-level decision.

8. **Population gaps as an equity issue.** African populations are essentially absent from training and evaluation. The screening gap that motivates this work concentrates in exactly the settings least represented in the data.

9. **Automation bias.** A calibrated probability presented to a non-specialist reader can anchor the reader's judgment, and the "Normal" label is especially prone to being read as "healthy." No human-factors evaluation of how this output affects reader behavior has been conducted.

---

## 12. Limitations and failure modes

Stated bluntly, and drawn from the manuscript's Limitations section and the OCT verification manifest.

1. **"Normal" means "no AMD/DME detected," not "healthy."** **347 of 354 eyes with genuine non-target pathology — including retinal artery occlusion and retinal vein occlusion — were output as "Normal."** There is no "other pathology / cannot classify" class. In screening use this is a false-reassurance hazard. The OOD flag rate within the confounder pool is not recorded, so it is unknown whether the gate would route these eyes to review.

2. **No validated strictly external DME estimate remains.** Noor and OCTID contain zero DME cases, and the archived Duke DME performance claim is withdrawn after the released-model reproduction gate failed. The DME class also received the least label-cleaning intervention (0 removals, per-class data-quality score 0.5938, probe recall 0.6296 below the 0.65 reliability gate).

3. **The OCTDL and Noor evaluations are NOT external.** Both were run on the **full** datasets, which include their training portions: 1,444 of 2,064 OCTDL images and 6,932 of 9,904 Noor images were used in training (both training counts report-stated). Their AUCs of 0.9990 and 0.9999 measure in-domain performance and **must not be presented as independent external validation.** The internal evaluation artifact lists only Kermany-Test as release-gate eligible.

4. **Duke evidence is withdrawn after a fail-closed reproduction failure.** The Srinivasan cohort has 3,231 scans nested within **45 volumes** (15/15/15). The historical pipeline collapsed them into three class-level units, and the released model did not reproduce the archived scan-level claim under any of six documented variants. No historical or replacement Duke scan-, DME-, OOD-, calibration-, or volume-level value is treated as evidence.

5. **CNV and drusen are collapsed into one AMD class, so referral urgency cannot be assigned.** The module cannot distinguish neovascular AMD (urgent anti-VEGF referral) from early or intermediate AMD (routine monitoring). External validation predominantly covers dry AMD, so zero-shot CNV detection is largely unmeasured.

6. **The Kermany-Test AUC has no stored confidence interval.** 0.9990075521010786 is reported without any CI anywhere in the artifacts. The CI [0.9989, 0.9996] that appears alongside an AUC of 0.999286 belongs to the **threshold partition** (n = 14,384), not to the held-out test set. A test-set CI must be computed from saved predictions.

7. **The internal split-integrity claim is unverified.** The "patient-level split, zero patient overlap" property of the custom Kermany re-split is **report-stated; no patient-overlap verification artifact exists.** The Kermany distribution is known to contain duplicate and near-duplicate B-scans and multiple images per patient, which have caused leakage in derived re-splits. If same-patient images straddle this split, the internal AUC and the calibration estimates are optimistically biased. The Kermany validation-split count (8,501) and the full source-to-partition accounting are also report-stated.

8. **Kermany-Test is a custom re-split, not the canonical benchmark.** It is an 11,129-image re-split, not the canonical 1,000-image four-class Kermany test set used by much of the published literature, so even a sanity-check comparison against that literature is approximate.

9. **The OOD gate is untested on the confounder pool and flags far more than the model misses.** Flag rates range from 4.67% to **66.63%** and correlate with device and acquisition characteristics rather than with observed error — discrimination is preserved on every flagged set. Two-thirds of Noor images are flagged even though roughly 70% of them were in the training set. A 38-67% flag rate would route one- to two-thirds of scans at a device-shifted site to human review.

10. **ONNX export parity is unverified.** The report's claimed parity differences (0.000287 probability, 0.000987 logit) have **no artifact backing**; `export_metadata.json` contains no verification-diff fields. Export parity is re-verification pending, not verified. No ONNX Runtime or edge-hardware benchmark has been run, and no artifact of the audited run recorded the export's file size.

11. **The 0.70 threshold has no documented clinical derivation, and its CI is degenerate.** It is a pre-specified cap (`max_clinical_threshold` = 0.7, `ceiling_applied` = true). Because the cap was imposed, every bootstrap resample returned 0.7, so the threshold CI is a single point with SD 0. The underlying selection rule (maximize specificity subject to sensitivity >= 0.90) is recovered from the released source, not stored in the artifacts (paper Section 2.4).

12. **Calibration drift under shift.** Held-out in-domain ECE is 0.019077097724760676 and stored OCTID ECE is 0.021204965024958167; the archived Duke calibration values are withdrawn. In-sample Venn-Abers ECE (0.000271) is a fitting diagnostic, not a performance claim. Venn-Abers guarantees hold under i.i.d. sampling and do not transfer to shifted prevalence; any target-site use would require local recalibration, which has not been validated.

13. **The Venn-Abers interval-to-scalar rule is absent from the run artifacts.** The rule is now specified from the released source (paper Section 2.4): the Vovk regularized point estimate p = p1 / (1 − p0 + p1), denominator floored at 1e-15, clipped to [0, 1]. The residual gap is that the deployed decision rule is evidenced by source code rather than by any artifact of the executed run.

14. **Unexplained evaluation-set attrition.** The report states 16,822 source images for Noor and 572 for OCTID; the evaluated counts are 9,904 and 261. **The attrition rule is not documented.** OCTID's excluded classes (macular hole, central serous retinopathy, diabetic retinopathy) are exactly the kind of confounder the stress test probes, and their exclusion has no recorded rationale.

15. **OCTID is a very small external set with no DME.** 261 images, **55 AMD positives**, zero DME cases. A single false negative produces its 98.18% sensitivity. Its stored macro AUC of 0.5 is degenerate and must never be cited, and no DME metric may be cited for it.

16. **Archived Duke evaluation artifacts are non-reproducible.** Multiple decision-rule and bootstrap outputs coexist, but none is retained as evidence after the released-model reproduction gate failed. OCTID has slightly different image-level AUC bounds in two artifacts; both should be interpreted cautiously because the cohort has only 55 AMD positives.

17. **Grad-CAM zone statistics and the RUN-010 comparators are report-stated with no artifact backing.** The 38.99% mean AMD-zone activation, the 6-of-50 below-threshold count, the 25% RUN-010 baseline, and the 14.1% RUN-010 false-AMD rate are stored in no readable artifact. The RUN-010 comparison is additionally uncontrolled (different task, data, and configuration), so **no causal attribution or improvement multiplier is claimed** for the spatial-attention regularizer or for the three-class redesign.

18. **The spatial-attention regularizer's anatomical assumption is only partly justified.** It penalizes activation outside rows 6-8 inclusive of the 15 x 15 feature map (the module report's "rows 6-9" quotes the exclusive slice bound), the outer-retina zone where drusen and RPE/Bruch's-complex changes predominate. **Neovascular disease may extend anterior to this band**, and consistent vertical centering of B-scans is not evidenced.

19. **Release-gate thresholds are report-asserted.** The pass/fail outcome is stored, and the achieved values are artifact-verified, but the numeric gate thresholds themselves are not stored in any evaluation artifact.

20. **Per-dataset CIs use 1,000 resamples, below the project's own convention.** The bootstrap protocol is recovered from the released source (percentile type, unstratified, fixed seeds; paper Section 2.7): per-dataset AUC/sensitivity/specificity CIs use 1,000 resamples (seeds 42/43/44) while operating-point CIs use 2,000. The 1,000-resample count falls short of the project's 2,000-resample reporting standard, and the counts are recorded in source rather than in the evaluation artifacts.

21. **Most of the training configuration is report-stated.** The 55,444 total training count, the Noor and OCTDL training counts, domain balancing, optimizer, learning rates, batch size, weight decay, EMA decay, focal gamma, MixUp alpha, class weights, patience, spatial-attention parameters, seed, parameter count, and GPU hardware have no artifact backing among the audited files.

22. **Single training run, single seed.** Run-to-run variance is unquantified; no repeat-run artifact exists.

23. **Per-B-scan, not per-patient.** All operating characteristics are per-B-scan. Clinical screening decisions are per-patient and integrate a full OCT volume. No valid volume-level evidence exists (item 4), and no per-patient aggregation rule is defined or validated.

24. **No gradability pathway.** All benchmarks are curated. The OOD gate measures feature-space distance, not scan quality, signal strength, or segmentation failure. Deployment would require an explicit ungradable pathway distinct from the OOD gate.

25. **Dataset terms and device attributions are external to the run artifacts.** No executed-run artifact records dataset licenses, device manufacturers, or source repositories. A separate source-page review on 2026-08-27 updated the summaries in Section 5, but learned-weight redistribution remains unestablished for several sources; Duke expressly limits use to research/education and prohibits redistribution and commercialization of the data.

26. **Retrospective only.** Every result is retrospective and computed on curated research datasets with varied access conditions. No prospective, intent-to-screen, or real-device validation exists. The documented retrospective-to-prospective compression in this field should be assumed to apply.

27. **No regulatory status.** RETGUARD has no regulatory clearance, authorization, or certification anywhere. It is not a medical device.
