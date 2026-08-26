# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Every user-facing string of the v1.1 interface, verbatim from UI_COPY.md.

This module is the single source of user-facing copy; ``ui.py`` never
contains a literal user-facing sentence. ``{placeholders}`` are ``str.format``
fields; the placeholder lists in UI_COPY.md are exhaustive.
"""

from typing import Final

PAGE_TITLE: Final = "RETGUARD Research Demo"

BANNER_MD: Final = (
    "**RETGUARD — research demonstration only.** Not a medical device. "
    "Not cleared or approved by any regulatory authority. Not for diagnosis, "
    "screening, triage, or any clinical decision about any patient."
)

ACK_TITLE: Final = "Before you start"

ACK_BODY_MD: Final = """\
RETGUARD is research software. Its outputs are model predictions on the submitted image, for research evaluation only — they are never medical advice and never a statement about any person's health.

- "Normal" from the OCT module means the model detected neither AMD nor DME. It does not mean the eye is healthy.
- On images unlike its training data, documented performance is below chance. The out-of-distribution warning exists because of this.
- Do not upload identifiable patient data, and do not upload a photo of your own eye seeking a diagnosis. The examples below the upload box mean you never need to upload anything of your own.
- If you have concerns about your eyes, see a qualified eye-care professional.\
"""

ACK_ARABIC: Final = (
    "هذا برنامج بحثي وليس جهازًا طبيًا، ولا يُستخدم للتشخيص. "
    "إذا كانت لديك مخاوف بشأن عينيك فاستشر طبيب عيون مختصًا."
)

ACK_BUTTON: Final = "I understand — research use only"

ANALYZE_DISABLED_HINT: Final = (
    'Read the note above and select "I understand" to enable analysis.'
)

TAB_DR: Final = "Diabetic retinopathy — fundus"

TAB_GLAUCOMA: Final = "Glaucoma — fundus"

TAB_OCT: Final = "AMD / DME — OCT B-scan"

UPLOAD_LABEL_FUNDUS: Final = "Color fundus photograph"

UPLOAD_LABEL_OCT: Final = "OCT B-scan"

UPLOAD_PLACEHOLDER_FUNDUS: Final = (
    "Drop a full color fundus photograph here (PNG or JPEG). The model requires "
    "a full fundus field — optic-disc crops are out of intended use."
)

UPLOAD_PLACEHOLDER_OCT: Final = "Drop a single OCT B-scan here (PNG or JPEG)."

UPLOAD_PHI_MD: Final = (
    "Do not upload identifiable patient data. Images are processed in memory and "
    "are not stored — see the privacy note in the footer. The examples below mean "
    "you never need to upload anything of your own."
)

UPLOAD_SIZE_NOTE: Final = "PNG or JPEG, up to 30 MB."

TTA_RADIO_LABEL: Final = "Inference protocol"

TTA_FAITHFUL_LABEL: Final = "Faithful — test-time augmentation (paper protocol)"

TTA_FAST_LABEL: Final = "Fast — single view"

TTA_FAST_CAPTION: Final = (
    "Single-view mode deviates from the published evaluation protocol, which "
    "averages logits over test-time augmentation views (8 for fundus, 2 for OCT). "
    "Calibration was fitted on TTA-averaged logits, so single-view probabilities "
    "can differ slightly from the published operating points. The "
    "out-of-distribution score uses the identity view only and is identical in "
    "both modes."
)

BUTTON_ANALYZE: Final = "Analyze"

BUTTON_CLEAR: Final = "Clear"

OOD_PASS_HTML: Final = (
    "IN DISTRIBUTION — this image resembles the module's training distribution "
    "(Mahalanobis score {score}, flag threshold {threshold}). The "
    "out-of-distribution check is advisory, not a guarantee."
)

OOD_FAIL_HTML: Final = (
    "OUT OF DISTRIBUTION — this image does not resemble the module's training "
    "data (Mahalanobis score {score}, flag threshold {threshold}). The prediction "
    "below is unreliable and is shown greyed out. On inputs like this, documented "
    "model performance is below chance."
)

OOD_UNRELIABLE_TAG: Final = "unreliable — out-of-distribution input"

DECISION_POSITIVE_DR: Final = "Model output: referable diabetic retinopathy"

DECISION_NEGATIVE_DR: Final = "Model output: no referable diabetic retinopathy detected"

DECISION_POSITIVE_GLAUCOMA: Final = "Model output: referable glaucoma"

DECISION_NEGATIVE_GLAUCOMA: Final = "Model output: no referable glaucoma detected"

DECISION_POSITIVE_OCT: Final = "Model output: AMD detected"

DECISION_NEGATIVE_OCT: Final = (
    "Model output: AMD not detected at the pre-specified threshold"
)

DECISION_SUBLINE: Final = (
    "Calibrated probability {probability} against the pre-specified threshold "
    "{threshold} ({threshold_provenance}; {card_section}). This is a model output "
    "on this image, not a referral and not medical advice."
)

THRESHOLD_PROVENANCE_VALIDATION: Final = "validation-derived"

THRESHOLD_PROVENANCE_OCT: Final = "a-priori engineering ceiling"

PROBABILITY_BAR_LABEL_DR: Final = (
    "Calibrated probability of referable DR (temperature-scaled)"
)

PROBABILITY_BAR_LABEL_GLAUCOMA: Final = (
    "Calibrated probability of referable glaucoma (Venn-Abers)"
)

PROBABILITY_BAR_LABEL_OCT: Final = "Calibrated probability of AMD (Venn-Abers)"

INTERVAL_LABEL: Final = "Venn-Abers interval {p0}–{p1}"

INTERVAL_MEANING: Final = (
    "The interval width is a per-input uncertainty signal: wider means the "
    "calibrated probability is less settled for this image."
)

DR_NO_INTERVAL_NOTE: Final = (
    "The DR module deploys temperature scaling only; it has no Venn-Abers interval."
)

DR_ECE_NOTE: Final = (
    "Disclosed limitation: DR calibration exceeds the project's own 0.05 "
    "expected-calibration-error gate (0.052537 on the calibration set; 0.0811 on "
    "Messidor-2). Treat the probability accordingly."
)

OCT_CLASS_BARS_LABEL: Final = (
    "Per-class breakdown — uncalibrated softmax over TTA mean logits"
)

OCT_DME_NOTE: Final = (
    "The deployed decision rule covers AMD only. The DME and Normal values above "
    "are uncalibrated model outputs with no deployed threshold."
)

NORMAL_NOT_HEALTHY_MD: Final = (
    '**"Normal" is not "healthy".** "Normal" means the model detected neither AMD '
    "nor DME — nothing more. In a 354-image confounder test, 347 of 354 other "
    "retinal pathologies (epiretinal membrane, retinal artery/vein occlusion, "
    "vitreomacular interface disease) were output as Normal."
)

CAM_CAPTION: Final = (
    "Left: the preprocessed view the model analyzed. Right: exact Grad-CAM "
    "computed from the shipped ONNX graph (identity view, last convolutional "
    "layer). The heatmap shows where the model's evidence concentrated — an "
    "attribution map, not clinical localization and not clinical evidence."
)

CAM_ALT: Final = (
    "Saliency heatmap overlaid on the preprocessed input; illustrative "
    "attribution map, not a clinical localization. The numeric prediction above "
    "carries the result."
)

CAM_WATERMARK: Final = "RESEARCH USE ONLY"

CAM_UNAVAILABLE_MD: Final = (
    "No saliency map is shown for this module. The exact-Grad-CAM verification "
    "did not pass for the installed model files, and RETGUARD does not display an "
    "attribution map it cannot verify."
)

RESULT_DISCLAIMER_MD: Final = (
    "Research demonstration only — not a medical device, and not a statement "
    "about any person's health. If you have concerns about your eyes, see a "
    "qualified eye-care professional."
)

PROVENANCE_LINE: Final = (
    "{latency_ms} ms · retguard 1.1.1 · {module} v1.0.0 · onnx sha256 "
    "{onnx_sha12} · CPU · {view_count}-view"
)

ACCORDION_DETAILS_LABEL: Final = "Model details and raw output"

DETAILS_BODY_MD: Final = (
    "Calibration: {calibration_method}. Threshold source: {threshold_source}. "
    "Full evaluation, limitations, and failure modes: [model card]({model_card_url})."
)

ACCORDION_ABOUT_LABEL: Final = "About this demo — intended use, evidence, limitations"

ABOUT_BODY_MD: Final = """\
RETGUARD is a research artifact by a single author (Sameh Aboelmaaty, Anchor Neuro). It is not a medical device, has no regulatory clearance in any jurisdiction, and must not be used for diagnosis, screening, triage, or patient management. All evidence is retrospective and based on research datasets with varied access conditions; no prospective validation and no validation on a target camera or OCT device has been performed.

External results (bootstrap 95% CI): DR — Messidor-2 (n=1,744, zero-shot) AUC 0.9691 (0.9595–0.9772), sensitivity 97.16%, specificity 76.07% at the pre-specified threshold. Glaucoma — REFUGE-Val (n=400, zero-shot) AUC 0.9278 (0.8642–0.9763); ORIGA (n=650, zero-shot) AUC 0.8729 (0.8399–0.9028). OCT — OCTID (n=261, external; 55 AMD positives and no DME cases) AMD-vs-rest AUC 0.9999 (0.9995–1.0000). The archived Duke performance claim is withdrawn after the released-model reproduction gate failed.

Known failure modes, reported with the same prominence: the glaucoma module performs below chance on optic-disc crops (RIM-ONE DL AUC 0.3667) and degrades on cropped fields of view; the OCT module outputs "Normal" for most non-AMD/DME pathologies (347 of 354 in a confounder test), and no validated strictly external DME performance estimate remains after withdrawal of the Duke claims; DR calibration exceeds the project's 0.05 ECE gate; one glaucoma result (ORIGA, AUC 0.9988 under training-domain preprocessing) is unconfirmed pending independent replication; the DR ONNX export failed its logit-parity tolerance (max difference 0.002526 vs 0.001), with probability parity passing at 0.000626.

Code: PolyForm Noncommercial 1.0.0. The release labels weights, model cards, and documentation CC BY-NC 4.0, but underlying dataset rights may impose additional limits; redistribution and commercial authorization are not established for every contributing source. Source, model cards, and citation: https://github.com/anchor-neuro/retguard\
"""

EMPTY_STATE_HTML: Final = (
    "Results will appear here: 1. distribution check (is this image like the "
    "training data), 2. model decision at the pre-specified threshold, "
    "3. calibrated probability, 4. saliency heatmap, 5. model details. Upload an "
    "image or pick an example to begin."
)

LOADING_TEXT_LOCAL: Final = (
    "Analyzing — {view_count}-view test-time augmentation on CPU, typically a "
    "few seconds."
)

LOADING_TEXT_HOSTED: Final = (
    "Analyzing — {view_count}-view test-time augmentation on shared CPU. "
    "Faithful mode can take 8 to 15 seconds here."
)

ERROR_NO_IMAGE: Final = "Upload an image first, or pick an example below the upload box."

ERROR_DECODE: Final = (
    "Could not decode this file as an image. PNG and JPEG are supported, up to 30 MB."
)

ERROR_IMAGE_SHAPE: Final = (
    "This image could not be processed: an 8-bit RGB image is required. Convert "
    "the file to standard PNG or JPEG and try again."
)

ERROR_NOT_ACKNOWLEDGED: Final = (
    'Select "I understand — research use only" at the top of the page before analyzing.'
)

EXAMPLE_LABEL_FIVES_DR: Final = "Fundus, FIVES dataset, CC BY 4.0 — not in DR training"

EXAMPLE_LABEL_FIVES_GLAUCOMA: Final = (
    "Fundus, FIVES dataset, CC BY 4.0 — dataset family seen in glaucoma training; "
    "expect optimistic output"
)

EXAMPLE_LABEL_KERMANY_AMD: Final = (
    "OCT B-scan (drusen/AMD), Kermany test split, CC BY 4.0 — held-out benchmark set"
)

EXAMPLE_LABEL_KERMANY_DME: Final = (
    "OCT B-scan (DME), Kermany test split, CC BY 4.0 — held-out benchmark set"
)

EXAMPLE_LABEL_KERMANY_NORMAL: Final = (
    "OCT B-scan (Normal), Kermany test split, CC BY 4.0 — held-out benchmark set"
)

EXAMPLE_LABEL_OOD: Final = (
    "Not a retinal image — demonstrates the out-of-distribution gate"
)

EXAMPLES_HEADER: Final = "Examples (licensed; attribution in the footer)"

ATTRIBUTION_FIVES: Final = (
    "Fundus photograph from FIVES (Jin K. et al., Sci Data 9:475, 2022), "
    "doi:10.6084/m9.figshare.19688169, licensed CC BY 4.0. Unmodified except resizing."
)

ATTRIBUTION_KERMANY: Final = (
    "OCT B-scan from the Kermany et al. dataset, test split (Kermany D. et al., "
    "Cell 172(5):1122-1131, 2018), Mendeley Data doi:10.17632/rscbjbr9sj.3, "
    "licensed CC BY 4.0. Unmodified except resizing."
)

ATTRIBUTION_OOD: Final = (
    "Photograph by Sameh Aboelmaaty, dedicated to the public domain (CC0). "
    "Included deliberately as a non-retinal input to demonstrate the "
    "out-of-distribution gate."
)

FOOTER_MD: Final = """\
For Research Use Only. Not for use in diagnostic procedures.

{privacy_md}

Code: PolyForm Noncommercial 1.0.0 · Weights: CC BY-NC 4.0 · Example images: attributions in ATTRIBUTIONS.md · [Intended use](https://github.com/anchor-neuro/retguard#intended-use-and-limitations) · [Model cards](https://github.com/anchor-neuro/retguard/tree/main/docs) · Source: https://github.com/anchor-neuro/retguard

Aboelmaaty S. RETGUARD: Retrospective Multi-Dataset Development and External Testing of Calibrated Deep-Learning Classifiers for Retinal Fundus Photographs and OCT B-Scans. Manuscript in preparation, 2026. · Sameh Aboelmaaty, Anchor Neuro — ghoneim2012@gmail.com\
"""

PRIVACY_LOCAL_MD: Final = (
    "Privacy: this server runs only on your machine (127.0.0.1) and makes no "
    "outbound network connections; no image leaves this machine. Uploads are "
    "processed in memory; temporary files are deleted within about a minute and "
    "nothing is stored after your session. Analytics and flagging are disabled."
)

PRIVACY_HOSTED_MD: Final = (
    "Privacy: uploaded images are processed transiently on this demo server and "
    "are not stored, logged, flagged, or used for training; temporary files are "
    "deleted within about a minute, and the server disk is wiped on every "
    "restart. Analytics and flagging are disabled. Hugging Face, the hosting "
    "platform, keeps standard web logs — see https://huggingface.co/privacy. Do "
    "not upload identifiable patient data. Data questions: ghoneim2012@gmail.com"
)

SERVE_TERMINAL_BANNER: Final = (
    "RETGUARD research demo — not a medical device; research use only."
)

SERVE_TERMINAL_URL: Final = "Serving at {url} (models loaded in {seconds} s)."

SERVE_TERMINAL_OFFLINE: Final = (
    "Fully offline: bound to this machine only; no image leaves it."
)

SERVE_TERMINAL_STOP: Final = "Press Ctrl+C to stop."

# The PyPI name "retguard" belongs to a third party, so the install
# instruction points at the GitHub tag, never at PyPI (binding
# install-command rule; UI_COPY.md 1.1 carries the same wording).
SERVE_MISSING_GRADIO: Final = (
    "The UI requires the optional dependency set: pip install "
    '"retguard[ui] @ git+https://github.com/anchor-neuro/retguard@v1.1.1"'
)

HUB_DISCLAIMER_MD: Final = """\
**Research demonstration only — not a medical device.** RETGUARD is not cleared or approved by any regulatory authority and must not be used for diagnosis, screening, triage, or any clinical decision about a real patient. Outputs are model predictions for research evaluation only. "Normal" means "no AMD/DME detected by the model," not "healthy." Performance on images outside the training domain is documented to be below chance. If you have concerns about your eyes, see a qualified eye-care professional.

For Research Use Only. Not for use in diagnostic procedures.\
"""
