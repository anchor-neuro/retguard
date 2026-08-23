# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Blocks construction, result-panel rendering, and copy-source discipline."""

import ast
from pathlib import Path

import pytest

from retguard import ui, ui_text
from retguard.constants import (
    DR_OOD_THRESHOLD,
    DR_THRESHOLD,
    GLAUCOMA_THRESHOLD,
    OCT_AMD_THRESHOLD,
)
from retguard.predictor import PredictionResult


def _result(
    module: str = "dr",
    probability: float = 0.612,
    decision: bool = True,
    threshold: float = DR_THRESHOLD,
    ood_score: float = 21.4,
    ood_flagged: bool = False,
    class_probabilities: dict[str, float] | None = None,
    venn_abers_interval: tuple[float, float] | None = None,
) -> PredictionResult:
    return PredictionResult(
        module=module,
        probability=probability,
        decision=decision,
        threshold=threshold,
        class_probabilities=class_probabilities,
        venn_abers_interval=venn_abers_interval,
        ood_score=ood_score,
        ood_flagged=ood_flagged,
    )


class _StubPredictor:
    module = "dr"
    threshold = DR_THRESHOLD


class TestBuildDemo:
    @pytest.fixture()
    def demo(self, monkeypatch: pytest.MonkeyPatch) -> object:
        monkeypatch.setattr(ui, "load", lambda module, weights_dir: _StubPredictor())
        monkeypatch.setattr(ui, "_cam_self_check", lambda predictor: True)
        return ui.build_demo()

    def test_analytics_disabled(self, demo: object) -> None:
        assert demo.analytics_enabled is False

    def test_delete_cache_purges_within_a_minute(self, demo: object) -> None:
        assert demo.delete_cache == (60, 60)

    def test_page_title_from_ui_text(self, demo: object) -> None:
        assert demo.title == ui_text.PAGE_TITLE

    def test_cam_failure_renders_the_unavailable_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ui, "load", lambda module, weights_dir: _StubPredictor())
        monkeypatch.setattr(ui, "_cam_self_check", lambda predictor: False)
        demo = ui.build_demo()
        rendered = [
            block.value
            for block in demo.blocks.values()
            if getattr(block, "value", None) == ui_text.CAM_UNAVAILABLE_MD
        ]
        assert len(rendered) == 3


class TestLaunchDemo:
    def test_launch_forwards_shared_theme_css_and_footer_links(self) -> None:
        """serve and the Space must launch identically (V11_BLUEPRINT 1.6)."""
        recorded: dict[str, object] = {}

        class _StubLaunchable:
            def launch(self, **kwargs: object) -> str:
                recorded.update(kwargs)
                return "launched"

        outcome = ui.launch_demo(_StubLaunchable(), server_port=None, quiet=True)
        assert outcome == "launched"
        assert recorded["theme"] is ui.THEME
        assert recorded["css"] is ui.CSS
        assert recorded["footer_links"] is ui._FOOTER_LINKS
        assert recorded["server_port"] is None
        assert recorded["quiet"] is True
        assert "share" not in recorded


class TestAcknowledgmentUpdates:
    def test_acknowledge_covers_panel_plus_all_buttons_and_hints(self) -> None:
        """One state, one panel, then per-module button and hint updates."""
        from retguard.constants import MODULES

        updates = ui._acknowledge()
        assert len(updates) == 2 + 2 * len(MODULES)
        assert updates[0] is True

    def test_hint_hidden_once_acknowledged_and_shown_before(self) -> None:
        """Tab select re-emits hint visibility (stale-hint regression)."""
        hidden = ui._hint_visibility(True)
        shown = ui._hint_visibility(False)
        assert hidden["visible"] is False  # type: ignore[index]
        assert shown["visible"] is True  # type: ignore[index]


class TestDecodeUpload:
    def test_undecodable_file_raises_the_typed_decode_error(
        self, tmp_path: Path
    ) -> None:
        """Regression: an undecodable upload must surface ERROR_DECODE, not
        gradio preprocess internals with local paths."""
        import gradio as gr

        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(gr.Error) as caught:
            ui._decode_upload(str(corrupt))
        assert ui_text.ERROR_DECODE in str(caught.value)

    def test_component_preprocess_raises_the_typed_decode_error(
        self, tmp_path: Path
    ) -> None:
        """gradio re-raises gr.Error from component preprocess untouched, so
        the toast shows ERROR_DECODE instead of the wrapped internal error."""
        import gradio as gr
        from gradio.data_classes import ImageData

        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        component = gr.Image(type="filepath", render=False)
        with pytest.raises(gr.Error) as caught:
            ui._guarded_image_preprocess(component, ImageData(path=str(corrupt)))
        assert ui_text.ERROR_DECODE in str(caught.value)

    def test_no_generated_component_stub_next_to_ui(self) -> None:
        """Subclassing a gradio component makes ComponentMeta write a
        generated ui.pyi into the package at import time; the upload guard
        must stay instance-level (_guarded_image_preprocess)."""
        assert not Path(ui.__file__).with_suffix(".pyi").exists()

    def test_valid_file_decodes_to_rgb_uint8(self, tmp_path: Path) -> None:
        import cv2
        import numpy as np

        path = tmp_path / "valid.png"
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[..., 0] = 255
        cv2.imwrite(str(path), bgr)
        decoded = ui._decode_upload(str(path))
        assert decoded.shape == (8, 8, 3)
        assert decoded.dtype == np.uint8
        assert int(decoded[0, 0, 2]) == 255


class TestRenderResultHtml:
    def test_dr_result_contains_decision_threshold_and_ood_text(self) -> None:
        html_out = ui._render_result_html(_result(), latency_ms=812, tta=True)
        assert ui_text.DECISION_POSITIVE_DR in html_out
        assert "0.204983" in html_out
        assert ui_text.THRESHOLD_PROVENANCE_VALIDATION in html_out
        assert "IN DISTRIBUTION" in html_out
        assert f"{DR_OOD_THRESHOLD:.1f}" in html_out
        assert ui_text.DR_NO_INTERVAL_NOTE in html_out
        assert ui_text.DR_ECE_NOTE in html_out

    def test_negative_decision_renders_the_negative_chip(self) -> None:
        html_out = ui._render_result_html(
            _result(probability=0.05, decision=False), latency_ms=10, tta=True
        )
        assert ui_text.DECISION_NEGATIVE_DR in html_out
        assert "retguard-chip-negative" in html_out

    def test_glaucoma_interval_rendered_on_the_bar(self) -> None:
        html_out = ui._render_result_html(
            _result(
                module="glaucoma",
                threshold=GLAUCOMA_THRESHOLD,
                venn_abers_interval=(0.101, 0.204),
            ),
            latency_ms=10,
            tta=True,
        )
        assert "Venn-Abers interval 0.101" in html_out
        assert ui_text.INTERVAL_MEANING in html_out
        assert "retguard-bar-interval" in html_out

    def test_oct_result_contains_class_bars_and_normal_not_healthy(self) -> None:
        html_out = ui._render_result_html(
            _result(
                module="oct",
                threshold=OCT_AMD_THRESHOLD,
                class_probabilities={"Normal": 0.2, "AMD": 0.7, "DME": 0.1},
                venn_abers_interval=(0.6, 0.8),
            ),
            latency_ms=10,
            tta=True,
        )
        assert ui_text.OCT_CLASS_BARS_LABEL in html_out
        assert ui_text.OCT_DME_NOTE in html_out
        assert "347 of 354" in html_out
        assert "<strong>&quot;Normal&quot; is not &quot;healthy&quot;.</strong>" in html_out
        # Digit-for-digit threshold display (paper writes 0.70, not 0.7), and
        # the honest provenance: the OCT threshold is a ceiling, not derived.
        assert "0.70" in html_out
        assert ui_text.THRESHOLD_PROVENANCE_OCT in html_out
        assert ui_text.THRESHOLD_PROVENANCE_VALIDATION not in html_out

    def test_ood_flagged_greys_sections_and_shows_fail_text(self) -> None:
        html_out = ui._render_result_html(
            _result(ood_score=90.6, ood_flagged=True), latency_ms=10, tta=True
        )
        assert "OUT OF DISTRIBUTION" in html_out
        assert "retguard-unreliable" in html_out
        assert ui_text.OOD_UNRELIABLE_TAG in html_out

    def test_dynamic_values_are_html_escaped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(ui._CARD_SECTIONS, "dr", '<script>alert("x")</script>')
        html_out = ui._render_result_html(_result(), latency_ms=10, tta=True)
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out


class TestProvenanceAndDetails:
    def test_provenance_line_fields(self) -> None:
        html_out = ui._render_provenance_html(_result(), latency_ms=1234, tta=True)
        assert "1234 ms" in html_out
        assert "retguard 1.1.0" in html_out
        assert "dr v1.0.0" in html_out
        assert "8-view" in html_out

    def test_single_view_provenance(self) -> None:
        html_out = ui._render_provenance_html(_result(), latency_ms=90, tta=False)
        assert "1-view" in html_out

    def test_details_markdown_contains_raw_json_and_card_link(self) -> None:
        markdown_out = ui._render_details_markdown(_result())
        assert "```json" in markdown_out
        assert '"ood_flagged": false' in markdown_out
        assert ui._MODEL_CARD_URLS["dr"] in markdown_out


class TestCopyDiscipline:
    def test_ui_module_contains_no_sentence_literals(self) -> None:
        """Every user-facing sentence must come from ui_text, not ui."""
        source_path = Path(ui.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        docstring_nodes = set()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    docstring_nodes.add(id(body[0].value))
            # The CSS stylesheet is markup, not user-facing prose.
            if isinstance(node, ast.AnnAssign) and (
                isinstance(node.target, ast.Name) and node.target.id == "CSS"
            ):
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant):
                        docstring_nodes.add(id(child))
        offending = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstring_nodes
            ):
                words = [
                    token
                    for token in node.value.replace("\n", " ").split(" ")
                    if token.isalpha()
                ]
                if len(words) >= 5:
                    offending.append(node.value)
        assert not offending, offending
