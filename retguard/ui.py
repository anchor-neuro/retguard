# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""The ``retguard serve`` and Hugging Face Space Blocks application.

One codebase, two deployments (V11_BLUEPRINT section 1.6): ``hosted`` only
switches the privacy notice. All user-facing copy comes from
:mod:`retguard.ui_text`; this module never contains a user-facing sentence.
"""

import dataclasses
import html
import json
import time
from collections.abc import Callable
from pathlib import Path
from types import MethodType
from typing import Any, Final, Literal, cast

import cv2
import gradio as gr
import numpy as np
import PIL.Image
from gradio.data_classes import ImageData

from retguard import ui_text
from retguard.cam import head_logits
from retguard.constants import (
    CAM_OVERLAY_ALPHA,
    DR_OOD_THRESHOLD,
    FUNDUS_TTA_VIEWS,
    GAP_OUTPUT_NODE,
    GLAUCOMA_OOD_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    MODULES,
    OCT_OOD_THRESHOLD,
    OCT_TTA_VIEWS,
    ONNX_INPUT_NAME,
    THRESHOLD_DISPLAY_DECIMALS,
)
from retguard.predictor import PredictionResult, Predictor, load
from retguard.preprocess import preprocess_dr, preprocess_glaucoma, preprocess_oct
from retguard.weights import RELEASE_TAG, WEIGHTS

THEME: Final = gr.themes.Soft()

# Gradio 6 moved theme/css to launch(); serve and the Space both go through
# launch_demo so the two deployments stay pixel-identical (research_demos §3).
CSS: Final = """
.retguard-banner {
  border: 1px solid var(--border-color-primary);
  border-inline-start: 4px solid #b45309;
  background: color-mix(in srgb, #b45309 12%, var(--block-background-fill));
  border-radius: 6px;
  padding: 10px 14px;
}
.retguard-banner p::before { content: "\\25B3\\00A0"; }
.retguard-empty {
  border: 1px dashed var(--border-color-primary);
  border-radius: 8px;
  padding: 24px 18px;
  color: var(--body-text-color);
}
.retguard-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 12px 14px;
  margin-block-end: 10px;
  background: var(--block-background-fill);
}
.retguard-ood-pass { border-inline-start: 4px solid var(--border-color-primary); }
.retguard-ood-fail {
  border-inline-start: 4px solid #b45309;
  background: color-mix(in srgb, #b45309 14%, var(--block-background-fill));
}
.retguard-ood-title::before { content: "\\25CF\\00A0"; }
.retguard-ood-fail .retguard-ood-title::before { content: "\\25B2\\00A0"; }
.retguard-chip {
  display: inline-block;
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
  padding: 4px 14px;
  font-weight: 600;
}
.retguard-chip-positive::before { content: "\\25CF\\00A0"; color: #b45309; }
.retguard-chip-negative::before { content: "\\25CB\\00A0"; }
/* Notes and sublines carry safety and limitation statements: full body text
   color, never the subdued token, which measures ~2.5:1 on white (WCAG AA
   needs 4.5:1 at these sizes). */
.retguard-subline { color: var(--body-text-color); margin-block-start: 6px; }
.retguard-bar-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 10px;
  align-items: center;
}
.retguard-bar-label { grid-column: 1 / -1; font-size: 0.9em; }
.retguard-bar-track {
  position: relative;
  height: 14px;
  border: 1px solid var(--border-color-primary);
  border-radius: 7px;
  background: color-mix(in srgb, var(--body-text-color) 6%, transparent);
  overflow: hidden;
}
.retguard-bar-fill {
  position: absolute;
  inset-block: 0;
  inset-inline-start: 0;
  background: color-mix(in srgb, var(--body-text-color) 55%, var(--block-background-fill));
}
.retguard-bar-interval {
  position: absolute;
  inset-block: 1px;
  border-inline: 2px solid #b45309;
  background: color-mix(in srgb, #b45309 25%, transparent);
}
.retguard-bar-value { font-variant-numeric: tabular-nums; font-weight: 600; }
.retguard-note { color: var(--body-text-color); font-size: 0.85em; margin-block-start: 6px; }
.retguard-tag {
  display: inline-block;
  border: 1px solid #b45309;
  border-radius: 4px;
  padding: 1px 8px;
  font-size: 0.8em;
  margin-block-end: 6px;
}
.retguard-unreliable { opacity: 0.45; }
/* OOD-flagged results tag the saliency pane (retguard-tag inside the wrap);
   grey the image pair with the same spec'd de-emphasis as the HTML cards. */
.retguard-saliency-wrap:has(.retguard-tag) .retguard-saliency-images {
  opacity: 0.45;
}
.retguard-provenance { color: var(--body-text-color); font-size: 0.8em; }
.retguard-rtl { direction: rtl; text-align: start; }
/* gradio 6.25 renders a stray "·" divider in its footer next to a hidden
   built-in button; with a single footer link there is nothing to divide. */
footer div.divider { display: none; }
/* 44 px minimum touch target (KG-6 checklist). */
.retguard-button { min-height: 44px; }
/* Example labels wrap instead of truncating: the honesty tag is the half a
   default-width column cuts off. */
.retguard-examples button,
.retguard-examples .gallery-item {
  white-space: normal;
  overflow-wrap: anywhere;
  height: auto;
}
.retguard-visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
}
@media (max-width: 700px) {
  .retguard-columns { flex-direction: column; }
}
"""

# Tolerance for the startup head-reproduces-logit self-check; the KG-3(a)
# exactness gate observed max error < 5e-7 on all three shipped graphs.
_CAM_SELF_CHECK_TOLERANCE: Final = 1e-4

_PREPROCESS: Final[dict[str, Callable[[np.ndarray], np.ndarray]]] = {
    "dr": preprocess_dr,
    "glaucoma": preprocess_glaucoma,
    "oct": preprocess_oct,
}
_TAB_LABELS: Final = {
    "dr": ui_text.TAB_DR,
    "glaucoma": ui_text.TAB_GLAUCOMA,
    "oct": ui_text.TAB_OCT,
}
_DECISION_POSITIVE: Final = {
    "dr": ui_text.DECISION_POSITIVE_DR,
    "glaucoma": ui_text.DECISION_POSITIVE_GLAUCOMA,
    "oct": ui_text.DECISION_POSITIVE_OCT,
}
_DECISION_NEGATIVE: Final = {
    "dr": ui_text.DECISION_NEGATIVE_DR,
    "glaucoma": ui_text.DECISION_NEGATIVE_GLAUCOMA,
    "oct": ui_text.DECISION_NEGATIVE_OCT,
}
_PROBABILITY_BAR_LABELS: Final = {
    "dr": ui_text.PROBABILITY_BAR_LABEL_DR,
    "glaucoma": ui_text.PROBABILITY_BAR_LABEL_GLAUCOMA,
    "oct": ui_text.PROBABILITY_BAR_LABEL_OCT,
}
_OOD_THRESHOLDS: Final = {
    "dr": DR_OOD_THRESHOLD,
    "glaucoma": GLAUCOMA_OOD_THRESHOLD,
    "oct": OCT_OOD_THRESHOLD,
}
_TTA_VIEW_COUNTS: Final = {
    "dr": FUNDUS_TTA_VIEWS,
    "glaucoma": FUNDUS_TTA_VIEWS,
    "oct": OCT_TTA_VIEWS,
}
# DECISION_SUBLINE / DETAILS_BODY_MD placeholder fills; each value points at
# the artifact that fixes the module's threshold (constants.py citations).
_CARD_SECTIONS: Final = {
    "dr": "paper section 2.4",
    "glaucoma": "paper section 2.4",
    "oct": "OCT model card section 8",
}
# The OCT 0.70 threshold is an a-priori engineering ceiling that bound in the
# executed run (paper section 2.4), never a validation-derived quantity.
_THRESHOLD_PROVENANCE: Final = {
    "dr": ui_text.THRESHOLD_PROVENANCE_VALIDATION,
    "glaucoma": ui_text.THRESHOLD_PROVENANCE_VALIDATION,
    "oct": ui_text.THRESHOLD_PROVENANCE_OCT,
}
_CALIBRATION_METHODS: Final = {
    "dr": "temperature scaling",
    "glaucoma": "Venn-Abers",
    "oct": "Venn-Abers",
}
_THRESHOLD_SOURCES: Final = {
    "dr": "external_eval_binary.json (paper section 2.4)",
    "glaucoma": "operating_point.json (paper section 2.4)",
    "oct": "max_clinical_threshold (OCT model card section 8)",
}
_MODEL_CARD_URLS: Final = {
    "dr": "https://github.com/anchor-neuro/retguard/blob/main/docs/MODEL_CARD_dr.md",
    "glaucoma": "https://github.com/anchor-neuro/retguard/blob/main/docs/MODEL_CARD_glaucoma.md",
    "oct": "https://github.com/anchor-neuro/retguard/blob/main/docs/MODEL_CARD_oct.md",
}
_EXAMPLE_FILES: Final = {
    "dr": (
        ("fives_dr_1.png", ui_text.EXAMPLE_LABEL_FIVES_DR),
        ("fives_dr_2.png", ui_text.EXAMPLE_LABEL_FIVES_DR),
        ("not_retinal.jpg", ui_text.EXAMPLE_LABEL_OOD),
    ),
    "glaucoma": (
        ("fives_glaucoma_1.png", ui_text.EXAMPLE_LABEL_FIVES_GLAUCOMA),
        ("fives_glaucoma_2.png", ui_text.EXAMPLE_LABEL_FIVES_GLAUCOMA),
        ("not_retinal.jpg", ui_text.EXAMPLE_LABEL_OOD),
    ),
    "oct": (
        ("kermany_amd.jpeg", ui_text.EXAMPLE_LABEL_KERMANY_AMD),
        ("kermany_dme.jpeg", ui_text.EXAMPLE_LABEL_KERMANY_DME),
        ("kermany_normal.jpeg", ui_text.EXAMPLE_LABEL_KERMANY_NORMAL),
        ("not_retinal.jpg", ui_text.EXAMPLE_LABEL_OOD),
    ),
}
# Custom {label, href} dicts are a silent no-op in the gradio 6.25 frontend
# (only the literal names render), so the intended-use and model-card links
# live in FOOTER_MD instead; "settings" keeps the theme toggle reachable.
_FOOTER_LINKS: Final[
    list[Literal["api", "gradio", "settings", "runs"] | dict[str, str]]
] = ["settings"]

# Watermark stamp geometry for the CAM overlay: gr.Image watermarks with an
# image, so CAM_WATERMARK is rasterized once onto a translucent plate.
_WATERMARK_FONT_SCALE: Final = 0.5
_WATERMARK_THICKNESS: Final = 1
_WATERMARK_PADDING_PX: Final = 6
_WATERMARK_PLATE_ALPHA: Final = 140


def _guarded_image_preprocess(
    component: gr.Image, payload: ImageData | None
) -> np.ndarray | PIL.Image.Image | str | None:
    """Map an undecodable upload to the typed ERROR_DECODE.

    gradio's own ``Image.preprocess`` opens the file with PIL and, on garbage
    input, raises before any handler runs - surfacing gradio internals with
    local filesystem paths. Probing with the same cv2 decode as
    ``preprocess.load_image`` turns that failure into the typed message and
    makes the UI accept exactly the formats the CLI accepts. gradio re-raises
    a ``gr.Error`` from component preprocess untouched. Bound per instance:
    subclassing a gradio component makes ComponentMeta write a generated
    ``ui.pyi`` into the installed package at import time.
    """
    if (
        payload is not None
        and payload.path is not None
        and cv2.imread(str(payload.path), cv2.IMREAD_COLOR) is None
    ):
        raise gr.Error(ui_text.ERROR_DECODE)
    return gr.Image.preprocess(component, payload)


def _watermark_stamp() -> np.ndarray:
    """Rasterize CAM_WATERMARK as an RGBA stamp for gr.Image's watermark."""
    (text_width, text_height), baseline = cv2.getTextSize(
        ui_text.CAM_WATERMARK,
        cv2.FONT_HERSHEY_SIMPLEX,
        _WATERMARK_FONT_SCALE,
        _WATERMARK_THICKNESS,
    )
    stamp = np.zeros(
        (
            text_height + baseline + 2 * _WATERMARK_PADDING_PX,
            text_width + 2 * _WATERMARK_PADDING_PX,
            4,
        ),
        dtype=np.uint8,
    )
    stamp[..., 3] = _WATERMARK_PLATE_ALPHA
    cv2.putText(
        stamp,
        ui_text.CAM_WATERMARK,
        (_WATERMARK_PADDING_PX, _WATERMARK_PADDING_PX + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        _WATERMARK_FONT_SCALE,
        (255, 255, 255, 255),
        _WATERMARK_THICKNESS,
        cv2.LINE_AA,
    )
    return stamp


def build_demo(
    weights_dir: Path | None = None,
    examples_dir: Path | None = None,
    hosted: bool = False,
) -> gr.Blocks:
    """Construct the research-demo Blocks app without launching it.

    Args:
        weights_dir: Directory with the extracted release artifacts. Defaults
            to ``$RETGUARD_WEIGHTS_DIR`` or ``~/.retguard``.
        examples_dir: Directory with the curated example images
            (V11_BLUEPRINT section 2.6); the gallery is omitted for any file
            that is absent.
        hosted: Render the Space privacy notice instead of the local one.

    Returns:
        The constructed :class:`gr.Blocks` application.
    """
    predictors = {module: load(module, weights_dir) for module in MODULES}
    cam_available = {
        module: _cam_self_check(predictors[module]) for module in MODULES
    }
    privacy_md = ui_text.PRIVACY_HOSTED_MD if hosted else ui_text.PRIVACY_LOCAL_MD

    with gr.Blocks(
        analytics_enabled=False,
        delete_cache=(60, 60),
        title=ui_text.PAGE_TITLE,
    ) as demo:
        gr.Markdown(ui_text.BANNER_MD, elem_classes=["retguard-banner"])
        acknowledged = gr.State(False)
        with gr.Group(visible=True) as acknowledgment_panel:
            gr.Markdown(f"### {ui_text.ACK_TITLE}")
            gr.Markdown(ui_text.ACK_BODY_MD)
            gr.Markdown(ui_text.ACK_ARABIC, elem_classes=["retguard-rtl"])
            acknowledge_button = gr.Button(
                ui_text.ACK_BUTTON,
                variant="primary",
                elem_classes=["retguard-button"],
            )

        analyze_buttons: list[gr.Button] = []
        hint_markdowns: list[gr.Markdown] = []
        modality_tabs: list[gr.Tab] = []
        for module in MODULES:
            with gr.Tab(_TAB_LABELS[module]) as modality_tab:
                analyze_button, hint_markdown = _build_tab(
                    module,
                    predictors[module],
                    cam_available[module],
                    examples_dir,
                    hosted,
                    acknowledged,
                )
            analyze_buttons.append(analyze_button)
            hint_markdowns.append(hint_markdown)
            modality_tabs.append(modality_tab)

        gr.Markdown(ui_text.FOOTER_MD.format(privacy_md=privacy_md))

        acknowledge_button.click(
            _acknowledge,
            inputs=None,
            outputs=[acknowledged, acknowledgment_panel, *analyze_buttons, *hint_markdowns],
            api_name="acknowledge",
        )
        # The acknowledge click's visibility updates reach only the mounted
        # (active) tab in gradio 6.25; re-emitting on tab select keeps the
        # hint state truthful on the other tabs.
        for module, modality_tab, hint_markdown in zip(
            MODULES, modality_tabs, hint_markdowns
        ):
            modality_tab.select(
                _hint_visibility,
                inputs=[acknowledged],
                outputs=[hint_markdown],
                api_name=f"hint_{module}",
            )
    return cast(gr.Blocks, demo)


# Any mirrors gr.Blocks.launch, whose kwargs and return are gradio-defined.
def launch_demo(demo: gr.Blocks, **kwargs: Any) -> Any:
    """Launch with the shared theme, CSS, and footer links (serve and Space)."""
    return demo.launch(theme=THEME, css=CSS, footer_links=_FOOTER_LINKS, **kwargs)


def _acknowledge() -> tuple[object, ...]:
    hide = gr.update(visible=False)
    enable = gr.update(interactive=True)
    return (True, hide, *([enable] * len(MODULES)), *([hide] * len(MODULES)))


def _hint_visibility(acknowledged: bool) -> object:
    return gr.update(visible=not acknowledged)


def _build_tab(
    module: str,
    predictor: Predictor,
    cam_ok: bool,
    examples_dir: Path | None,
    hosted: bool,
    acknowledged: gr.State,
) -> tuple[gr.Button, gr.Markdown]:
    """One modality tab: input column, result column, and their wiring."""
    is_oct = module == "oct"
    with gr.Row(elem_classes=["retguard-columns"]):
        with gr.Column(scale=1):
            # type="filepath" plus the preprocess guard: the handler decodes
            # with the CLI's own loader, and an undecodable upload surfaces
            # the typed ERROR_DECODE instead of gradio internals.
            image_input = gr.Image(
                type="filepath",
                sources=["upload", "clipboard"],
                height=320,
                label=ui_text.UPLOAD_LABEL_OCT if is_oct else ui_text.UPLOAD_LABEL_FUNDUS,
                placeholder=(
                    ui_text.UPLOAD_PLACEHOLDER_OCT
                    if is_oct
                    else ui_text.UPLOAD_PLACEHOLDER_FUNDUS
                ),
            )
            image_input.preprocess = MethodType(  # type: ignore[method-assign]
                _guarded_image_preprocess, image_input
            )
            gr.Markdown(ui_text.UPLOAD_PHI_MD)
            gr.Markdown(ui_text.UPLOAD_SIZE_NOTE, elem_classes=["retguard-note"])
            protocol_radio = gr.Radio(
                choices=[ui_text.TTA_FAITHFUL_LABEL, ui_text.TTA_FAST_LABEL],
                value=ui_text.TTA_FAITHFUL_LABEL,
                label=ui_text.TTA_RADIO_LABEL,
            )
            fast_caption = gr.Markdown(
                ui_text.TTA_FAST_CAPTION,
                visible=False,
                elem_classes=["retguard-note"],
            )
            with gr.Row():
                clear_button = gr.Button(
                    ui_text.BUTTON_CLEAR,
                    variant="secondary",
                    elem_classes=["retguard-button"],
                )
                analyze_button = gr.Button(
                    ui_text.BUTTON_ANALYZE,
                    variant="primary",
                    interactive=False,
                    elem_classes=["retguard-button"],
                )
            hint_markdown = gr.Markdown(
                ui_text.ANALYZE_DISABLED_HINT, elem_classes=["retguard-note"]
            )
            example_rows = _existing_examples(module, examples_dir)
            if example_rows:
                with gr.Column(elem_classes=["retguard-examples"]):
                    # gr.Examples renders caption-only buttons (no
                    # thumbnails) when example_labels is set in gradio 6.25,
                    # so a captioned gallery shows the images and clicking a
                    # tile loads that file into the input. Tile captions may
                    # truncate; the full labels render below in tile order.
                    example_gallery = gr.Gallery(
                        value=[(path, label) for path, label in example_rows],
                        label=ui_text.EXAMPLES_HEADER,
                        columns=min(len(example_rows), 4),
                        rows=1,
                        height=150,
                        allow_preview=False,
                        buttons=[],
                        interactive=False,
                    )

                    def _load_example(
                        evt: gr.SelectData,
                        rows: tuple[tuple[str, str], ...] = tuple(example_rows),
                    ) -> str:
                        index = evt.index
                        if not isinstance(index, int):
                            raise TypeError(
                                f"unexpected gallery selection index: {index!r}"
                            )
                        return rows[index][0]

                    example_gallery.select(_load_example, outputs=[image_input])
                    gr.Markdown(
                        "\n".join(
                            f"{index}. {label}"
                            for index, (_, label) in enumerate(example_rows, start=1)
                        ),
                        elem_classes=["retguard-note"],
                    )
        with gr.Column(scale=2):
            result_html = gr.HTML(_empty_state_html())
            with gr.Column(visible=False) as result_extras:
                cam_image: gr.Image | None = None
                with gr.Column(elem_classes=["retguard-saliency-wrap"]):
                    saliency_tag = gr.HTML("")
                    gr.HTML(
                        f'<span class="retguard-visually-hidden">'
                        f"{html.escape(ui_text.CAM_ALT)}</span>"
                    )
                    with gr.Row(elem_classes=["retguard-saliency-images"]):
                        preprocessed_image = gr.Image(
                            type="numpy", interactive=False, show_label=False
                        )
                        if cam_ok:
                            cam_image = gr.Image(
                                type="numpy",
                                interactive=False,
                                show_label=False,
                                watermark=gr.WatermarkOptions(
                                    watermark=_watermark_stamp(),
                                    position="bottom-right",
                                ),
                            )
                if cam_ok:
                    gr.Markdown(ui_text.CAM_CAPTION, elem_classes=["retguard-note"])
                else:
                    gr.Markdown(ui_text.CAM_UNAVAILABLE_MD)
                gr.Markdown(ui_text.RESULT_DISCLAIMER_MD)
                provenance_html = gr.HTML("")
                with gr.Accordion(ui_text.ACCORDION_DETAILS_LABEL, open=False):
                    details_markdown = gr.Markdown("")
            with gr.Accordion(ui_text.ACCORDION_ABOUT_LABEL, open=False):
                gr.Markdown(ui_text.ABOUT_BODY_MD)

    analyze_outputs: list[Any] = [
        result_html,
        result_extras,
        saliency_tag,
        preprocessed_image,
        provenance_html,
        details_markdown,
    ]
    if cam_image is not None:
        analyze_outputs.append(cam_image)
    analyze_button.click(
        _make_analyze_handler(module, predictor, cam_ok, hosted),
        inputs=[image_input, protocol_radio, acknowledged],
        outputs=analyze_outputs,
        api_name=f"analyze_{module}",
    )
    clear_button.click(
        _make_clear_handler(cam_ok),
        inputs=None,
        outputs=[image_input, *analyze_outputs],
        api_name=f"clear_{module}",
    )
    protocol_radio.change(
        _fast_caption_visibility,
        inputs=[protocol_radio],
        outputs=[fast_caption],
        api_name=f"protocol_{module}",
    )
    return analyze_button, hint_markdown


def _make_analyze_handler(
    module: str, predictor: Predictor, cam_ok: bool, hosted: bool
) -> Callable[..., tuple[object, ...]]:
    def analyze(
        image_path: str | None,
        protocol: str,
        acknowledged: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008 - gradio injects via the default
    ) -> tuple[object, ...]:
        if not acknowledged:
            raise gr.Error(ui_text.ERROR_NOT_ACKNOWLEDGED)
        if image_path is None:
            raise gr.Error(ui_text.ERROR_NO_IMAGE)
        image = _decode_upload(image_path)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise gr.Error(ui_text.ERROR_IMAGE_SHAPE)
        tta = protocol != ui_text.TTA_FAST_LABEL
        view_count = _TTA_VIEW_COUNTS[module] if tta else 1
        loading_template = (
            ui_text.LOADING_TEXT_HOSTED if hosted else ui_text.LOADING_TEXT_LOCAL
        )
        progress(0, desc=loading_template.format(view_count=view_count))

        started = time.perf_counter()
        heatmap: np.ndarray | None = None
        if cam_ok:
            result, heatmap = predictor.predict_with_cam(image, tta=tta)
        else:
            result = predictor.predict(image, tta=tta)
        latency_ms = round((time.perf_counter() - started) * 1000.0)

        display_view = _display_view(_PREPROCESS[module](image))
        outputs: list[object] = [
            _render_result_html(result, latency_ms, tta),
            gr.update(visible=True),
            _render_saliency_tag(result),
            display_view,
            _render_provenance_html(result, latency_ms, tta),
            _render_details_markdown(result),
        ]
        if cam_ok:
            assert heatmap is not None
            outputs.append(_overlay_heatmap(display_view, heatmap))
        return tuple(outputs)

    return analyze


def _make_clear_handler(cam_ok: bool) -> Callable[[], tuple[object, ...]]:
    def clear() -> tuple[object, ...]:
        outputs: list[object] = [
            None,
            _empty_state_html(),
            gr.update(visible=False),
            "",
            None,
            "",
            "",
        ]
        if cam_ok:
            outputs.append(None)
        return tuple(outputs)

    return clear


def _fast_caption_visibility(protocol: str) -> object:
    return gr.update(visible=protocol == ui_text.TTA_FAST_LABEL)


def _decode_upload(image_path: str) -> np.ndarray:
    """Decode an uploaded file exactly as ``preprocess.load_image`` does.

    cv2.imread returns ``None`` instead of raising, so the undecodable case
    maps to the typed ``ERROR_DECODE`` without an exception handler.
    """
    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise gr.Error(ui_text.ERROR_DECODE)
    return np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def _cam_self_check(predictor: Predictor) -> bool:
    """Startup gate: the numpy head must reproduce the graph logit on zeros."""
    zeros = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    logit, pooled = predictor._session.run(
        ["logit", GAP_OUTPUT_NODE[predictor.module]], {ONNX_INPUT_NAME: zeros}
    )
    reproduced = head_logits(pooled, predictor._cam_head)
    max_error = float(
        np.max(np.abs(reproduced.ravel() - np.asarray(logit).ravel()))
    )
    return max_error < _CAM_SELF_CHECK_TOLERANCE


def _display_view(tensor: np.ndarray) -> np.ndarray:
    """Invert ImageNet normalization to recover the uint8 view the model saw."""
    rgb = tensor.transpose(1, 2, 0) * (
        np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    ) + np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    view: np.ndarray = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return view


def _overlay_heatmap(display_view: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend the turbo-colored heatmap over the preprocessed view."""
    colored = cv2.applyColorMap(
        np.rint(heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (
        (1.0 - CAM_OVERLAY_ALPHA) * display_view.astype(np.float32)
        + CAM_OVERLAY_ALPHA * colored.astype(np.float32)
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _existing_examples(
    module: str, examples_dir: Path | None
) -> list[tuple[str, str]]:
    if examples_dir is None:
        return []
    rows = []
    for filename, label in _EXAMPLE_FILES[module]:
        path = examples_dir / filename
        if path.is_file():
            rows.append((str(path), label))
    return rows


def _empty_state_html() -> str:
    return f'<div class="retguard-empty">{ui_text.EMPTY_STATE_HTML}</div>'


def _render_saliency_tag(result: PredictionResult) -> str:
    if not result.ood_flagged:
        return ""
    return f'<span class="retguard-tag">{ui_text.OOD_UNRELIABLE_TAG}</span>'


def _render_result_html(result: PredictionResult, latency_ms: int, tta: bool) -> str:
    """The structured result panel: OOD card, decision chip, probability bars.

    Pure function of the prediction; unit-testable without a server. All
    interpolated values are HTML-escaped.
    """
    sections = [_ood_card_html(result)]
    tag = (
        f'<span class="retguard-tag">{ui_text.OOD_UNRELIABLE_TAG}</span>'
        if result.ood_flagged
        else ""
    )
    guarded_class = " retguard-unreliable" if result.ood_flagged else ""
    decision_text = (
        _DECISION_POSITIVE[result.module]
        if result.decision
        else _DECISION_NEGATIVE[result.module]
    )
    chip_class = (
        "retguard-chip-positive" if result.decision else "retguard-chip-negative"
    )
    threshold_decimals = THRESHOLD_DISPLAY_DECIMALS[result.module]
    subline = ui_text.DECISION_SUBLINE.format(
        probability=html.escape(f"{result.probability:.3f}"),
        threshold=html.escape(f"{result.threshold:.{threshold_decimals}f}"),
        threshold_provenance=html.escape(_THRESHOLD_PROVENANCE[result.module]),
        card_section=html.escape(_CARD_SECTIONS[result.module]),
    )
    sections.append(
        f'<div class="retguard-card{guarded_class}">{tag}'
        f'<div><span class="retguard-chip {chip_class}">'
        f"{decision_text}</span></div>"
        f'<div class="retguard-subline">{subline}</div></div>'
    )
    sections.append(
        f'<div class="retguard-card{guarded_class}">{tag}'
        f"{_probability_bar_html(result)}</div>"
    )
    if result.module == "oct" and result.class_probabilities is not None:
        class_bars = "".join(
            _bar_html(html.escape(name), value)
            for name, value in result.class_probabilities.items()
        )
        sections.append(
            f'<div class="retguard-card{guarded_class}">{tag}'
            f'<div class="retguard-bar-label">'
            f"{ui_text.OCT_CLASS_BARS_LABEL}</div>{class_bars}"
            f'<div class="retguard-note">{ui_text.OCT_DME_NOTE}</div>'
            f"</div>"
        )
        # Safety copy stays at full prominence even when the OOD gate greys
        # the prediction sections (V11_BLUEPRINT §2.5.4).
        sections.append(
            f'<div class="retguard-card">'
            f"{_markdown_bold_as_html(ui_text.NORMAL_NOT_HEALTHY_MD)}</div>"
        )
    return "".join(sections)


def _markdown_bold_as_html(markdown_text: str) -> str:
    """Escape a Markdown string and convert its ``**bold**`` pairs to HTML."""
    fragments = html.escape(markdown_text).split("**")
    return "".join(
        fragment if index % 2 == 0 else f"<strong>{fragment}</strong>"
        for index, fragment in enumerate(fragments)
    )


def _ood_card_html(result: PredictionResult) -> str:
    template = ui_text.OOD_FAIL_HTML if result.ood_flagged else ui_text.OOD_PASS_HTML
    state_class = "retguard-ood-fail" if result.ood_flagged else "retguard-ood-pass"
    body = template.format(
        score=html.escape(f"{result.ood_score:.1f}"),
        threshold=html.escape(f"{_OOD_THRESHOLDS[result.module]:.1f}"),
    )
    return (
        f'<div class="retguard-card {state_class}">'
        f'<span class="retguard-ood-title"></span>{body}</div>'
    )


def _probability_bar_html(result: PredictionResult) -> str:
    label = _PROBABILITY_BAR_LABELS[result.module]
    parts = [f'<div class="retguard-bar-label">{label}</div>']
    interval = result.venn_abers_interval
    parts.append(_bar_html("", result.probability, interval))
    if result.module == "dr":
        parts.append(
            f'<div class="retguard-note">'
            f"{ui_text.DR_NO_INTERVAL_NOTE}</div>"
            f'<div class="retguard-note">{ui_text.DR_ECE_NOTE}</div>'
        )
    elif interval is not None:
        interval_label = ui_text.INTERVAL_LABEL.format(
            p0=html.escape(f"{interval[0]:.3f}"), p1=html.escape(f"{interval[1]:.3f}")
        )
        parts.append(
            f'<div class="retguard-note">{interval_label} — '
            f"{ui_text.INTERVAL_MEANING}</div>"
        )
    return "".join(parts)


def _bar_html(
    label: str,
    value: float,
    interval: tuple[float, float] | None = None,
) -> str:
    fill_percent = max(0.0, min(1.0, value)) * 100.0
    interval_div = ""
    if interval is not None:
        start_percent = max(0.0, min(1.0, interval[0])) * 100.0
        width_percent = max(0.0, min(1.0, interval[1]) - max(0.0, interval[0])) * 100.0
        interval_div = (
            f'<div class="retguard-bar-interval" '
            f'style="inset-inline-start:{start_percent:.2f}%;'
            f'width:{width_percent:.2f}%"></div>'
        )
    label_div = f'<div class="retguard-bar-label">{label}</div>' if label else ""
    return (
        f'{label_div}<div class="retguard-bar-row">'
        f'<div class="retguard-bar-track">'
        f'<div class="retguard-bar-fill" style="width:{fill_percent:.2f}%"></div>'
        f"{interval_div}</div>"
        f'<div class="retguard-bar-value">{value:.3f}</div></div>'
    )


def _render_provenance_html(
    result: PredictionResult, latency_ms: int, tta: bool
) -> str:
    onnx_name = f"retguard_{result.module}_{RELEASE_TAG}.onnx"
    onnx_sha12 = WEIGHTS[result.module].files[onnx_name][:12]
    view_count = _TTA_VIEW_COUNTS[result.module] if tta else 1
    line = ui_text.PROVENANCE_LINE.format(
        latency_ms=html.escape(str(latency_ms)),
        module=html.escape(result.module),
        onnx_sha12=html.escape(onnx_sha12),
        view_count=html.escape(str(view_count)),
    )
    return f'<div class="retguard-provenance">{line}</div>'


def _render_details_markdown(result: PredictionResult) -> str:
    body = ui_text.DETAILS_BODY_MD.format(
        calibration_method=_CALIBRATION_METHODS[result.module],
        threshold_source=_THRESHOLD_SOURCES[result.module],
        model_card_url=_MODEL_CARD_URLS[result.module],
    )
    raw_json = json.dumps(dataclasses.asdict(result), indent=2)
    return f"{body}\n\n```json\n{raw_json}\n```"
