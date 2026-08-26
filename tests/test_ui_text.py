# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""UI copy completeness and register: every key present, safety strings verbatim.

The expected keys and safety sentences are duplicated here from UI_COPY.md on
purpose: the copy was reviewed once and is binding, so any drift in
``ui_text.py`` must fail a test rather than pass silently.
"""

import re
import string
from pathlib import Path

import pytest

from retguard import ui_text

_REPO_ROOT = Path(__file__).parents[1]

_EXPECTED_KEYS = (
    "PAGE_TITLE",
    "BANNER_MD",
    "ACK_TITLE",
    "ACK_BODY_MD",
    "ACK_ARABIC",
    "ACK_BUTTON",
    "ANALYZE_DISABLED_HINT",
    "TAB_DR",
    "TAB_GLAUCOMA",
    "TAB_OCT",
    "UPLOAD_LABEL_FUNDUS",
    "UPLOAD_LABEL_OCT",
    "UPLOAD_PLACEHOLDER_FUNDUS",
    "UPLOAD_PLACEHOLDER_OCT",
    "UPLOAD_PHI_MD",
    "UPLOAD_SIZE_NOTE",
    "TTA_RADIO_LABEL",
    "TTA_FAITHFUL_LABEL",
    "TTA_FAST_LABEL",
    "TTA_FAST_CAPTION",
    "BUTTON_ANALYZE",
    "BUTTON_CLEAR",
    "OOD_PASS_HTML",
    "OOD_FAIL_HTML",
    "OOD_UNRELIABLE_TAG",
    "DECISION_POSITIVE_DR",
    "DECISION_NEGATIVE_DR",
    "DECISION_POSITIVE_GLAUCOMA",
    "DECISION_NEGATIVE_GLAUCOMA",
    "DECISION_POSITIVE_OCT",
    "DECISION_NEGATIVE_OCT",
    "DECISION_SUBLINE",
    "THRESHOLD_PROVENANCE_VALIDATION",
    "THRESHOLD_PROVENANCE_OCT",
    "PROBABILITY_BAR_LABEL_DR",
    "PROBABILITY_BAR_LABEL_GLAUCOMA",
    "PROBABILITY_BAR_LABEL_OCT",
    "INTERVAL_LABEL",
    "INTERVAL_MEANING",
    "DR_NO_INTERVAL_NOTE",
    "DR_ECE_NOTE",
    "OCT_CLASS_BARS_LABEL",
    "OCT_DME_NOTE",
    "NORMAL_NOT_HEALTHY_MD",
    "CAM_CAPTION",
    "CAM_ALT",
    "CAM_WATERMARK",
    "CAM_UNAVAILABLE_MD",
    "RESULT_DISCLAIMER_MD",
    "PROVENANCE_LINE",
    "ACCORDION_DETAILS_LABEL",
    "DETAILS_BODY_MD",
    "ACCORDION_ABOUT_LABEL",
    "ABOUT_BODY_MD",
    "EMPTY_STATE_HTML",
    "LOADING_TEXT_LOCAL",
    "LOADING_TEXT_HOSTED",
    "ERROR_NO_IMAGE",
    "ERROR_DECODE",
    "ERROR_IMAGE_SHAPE",
    "ERROR_NOT_ACKNOWLEDGED",
    "EXAMPLE_LABEL_FIVES_DR",
    "EXAMPLE_LABEL_FIVES_GLAUCOMA",
    "EXAMPLE_LABEL_KERMANY_AMD",
    "EXAMPLE_LABEL_KERMANY_DME",
    "EXAMPLE_LABEL_KERMANY_NORMAL",
    "EXAMPLE_LABEL_OOD",
    "EXAMPLES_HEADER",
    "ATTRIBUTION_FIVES",
    "ATTRIBUTION_KERMANY",
    "ATTRIBUTION_OOD",
    "FOOTER_MD",
    "PRIVACY_LOCAL_MD",
    "PRIVACY_HOSTED_MD",
    "SERVE_TERMINAL_BANNER",
    "SERVE_TERMINAL_URL",
    "SERVE_TERMINAL_OFFLINE",
    "SERVE_TERMINAL_STOP",
    "SERVE_MISSING_GRADIO",
    "HUB_DISCLAIMER_MD",
)

# UI_COPY.md documents the placeholder list under each parameterized string
# as exhaustive; every other string must contain no format field at all.
_EXPECTED_PLACEHOLDERS = {
    "OOD_PASS_HTML": {"score", "threshold"},
    "OOD_FAIL_HTML": {"score", "threshold"},
    "DECISION_SUBLINE": {
        "probability",
        "threshold",
        "threshold_provenance",
        "card_section",
    },
    "INTERVAL_LABEL": {"p0", "p1"},
    "PROVENANCE_LINE": {"latency_ms", "module", "onnx_sha12", "view_count"},
    "DETAILS_BODY_MD": {"calibration_method", "threshold_source", "model_card_url"},
    "LOADING_TEXT_LOCAL": {"view_count"},
    "LOADING_TEXT_HOSTED": {"view_count"},
    "FOOTER_MD": {"privacy_md"},
    "SERVE_TERMINAL_URL": {"url", "seconds"},
}

# Verbatim safety copy from UI_COPY.md; a builder may never reword these.
_VERBATIM_SAFETY_STRINGS = {
    "BANNER_MD": (
        "**RETGUARD — research demonstration only.** Not a medical device. "
        "Not cleared or approved by any regulatory authority. Not for "
        "diagnosis, screening, triage, or any clinical decision about any patient."
    ),
    "ACK_BUTTON": "I understand — research use only",
    "ACK_ARABIC": (
        "هذا برنامج بحثي وليس جهازًا طبيًا، ولا يُستخدم للتشخيص. "
        "إذا كانت لديك مخاوف بشأن عينيك فاستشر طبيب عيون مختصًا."
    ),
    "RESULT_DISCLAIMER_MD": (
        "Research demonstration only — not a medical device, and not a "
        "statement about any person's health. If you have concerns about your "
        "eyes, see a qualified eye-care professional."
    ),
    "NORMAL_NOT_HEALTHY_MD": (
        '**"Normal" is not "healthy".** "Normal" means the model detected '
        "neither AMD nor DME — nothing more. In a 354-image confounder test, "
        "347 of 354 other retinal pathologies (epiretinal membrane, retinal "
        "artery/vein occlusion, vitreomacular interface disease) were output "
        "as Normal."
    ),
    "OOD_FAIL_HTML": (
        "OUT OF DISTRIBUTION — this image does not resemble the module's "
        "training data (Mahalanobis score {score}, flag threshold "
        "{threshold}). The prediction below is unreliable and is shown greyed "
        "out. On inputs like this, documented model performance is below chance."
    ),
    "CAM_WATERMARK": "RESEARCH USE ONLY",
    "CAM_UNAVAILABLE_MD": (
        "No saliency map is shown for this module. The exact-Grad-CAM "
        "verification did not pass for the installed model files, and "
        "RETGUARD does not display an attribution map it cannot verify."
    ),
}

# The RUO formula must appear letter-for-letter wherever it is rendered.
_RUO_FORMULA = "For Research Use Only. Not for use in diagnostic procedures."

# Copy the package repo never renders: staged into the Space and Hub repos.
_DEPLOY_ONLY_KEYS = {
    "ATTRIBUTION_FIVES",
    "ATTRIBUTION_KERMANY",
    "ATTRIBUTION_OOD",
    "HUB_DISCLAIMER_MD",
}

_EMOJI_PATTERN = re.compile(
    "[\U0001f000-\U0001fbff☀-➿️←-⇿⬀-⯿]"
)

_SAMPLE_FILL = {
    "score": "12.3",
    "threshold": "47.4",
    "probability": "0.612",
    "threshold_provenance": "validation-derived",
    "card_section": "model card section 8",
    "p0": "0.101",
    "p1": "0.204",
    "latency_ms": "812",
    "module": "dr",
    "onnx_sha12": "f0e19fa86d5a",
    "view_count": "8",
    "calibration_method": "temperature scaling",
    "threshold_source": "operating_point.json",
    "model_card_url": "https://github.com/anchor-neuro/retguard",
    "privacy_md": "privacy",
    "url": "http://127.0.0.1:7860/",
    "seconds": "6.1",
}


def _all_strings() -> dict[str, str]:
    return {key: getattr(ui_text, key) for key in _EXPECTED_KEYS}


def _placeholders(template: str) -> set[str]:
    return {
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field is not None
    }


class TestCompleteness:
    def test_every_ui_copy_key_is_a_string_constant(self) -> None:
        missing = [
            key
            for key in _EXPECTED_KEYS
            if not isinstance(getattr(ui_text, key, None), str)
        ]
        assert not missing, missing

    def test_no_undocumented_public_constant(self) -> None:
        """ui_text carries exactly the UI_COPY.md keys, nothing improvised."""
        public = {
            name
            for name in vars(ui_text)
            if name.isupper() and isinstance(getattr(ui_text, name), str)
        }
        assert public == set(_EXPECTED_KEYS)

    def test_every_rendered_constant_is_referenced_by_ui_or_cli(self) -> None:
        rendered_source = (
            (_REPO_ROOT / "retguard" / "ui.py").read_text(encoding="utf-8")
            + (_REPO_ROOT / "retguard" / "cli.py").read_text(encoding="utf-8")
        )
        orphaned = [
            key
            for key in _EXPECTED_KEYS
            if key not in _DEPLOY_ONLY_KEYS and key not in rendered_source
        ]
        assert not orphaned, orphaned


class TestSafetyStringsVerbatim:
    @pytest.mark.parametrize("key", sorted(_VERBATIM_SAFETY_STRINGS))
    def test_safety_string_matches_ui_copy_byte_for_byte(self, key: str) -> None:
        assert getattr(ui_text, key) == _VERBATIM_SAFETY_STRINGS[key]

    def test_ruo_formula_verbatim_in_footer_and_hub_disclaimer(self) -> None:
        assert ui_text.FOOTER_MD.startswith(_RUO_FORMULA)
        assert _RUO_FORMULA in ui_text.HUB_DISCLAIMER_MD

    def test_ack_body_states_below_chance_and_see_a_professional(self) -> None:
        assert "documented performance is below chance" in ui_text.ACK_BODY_MD
        assert "see a qualified eye-care professional" in ui_text.ACK_BODY_MD
        assert "It does not mean the eye is healthy" in ui_text.ACK_BODY_MD

    def test_privacy_notices_never_overclaim(self) -> None:
        """Blueprint 5.3: 'deleted within about a minute', never 'never touches disk'."""
        for notice in (ui_text.PRIVACY_LOCAL_MD, ui_text.PRIVACY_HOSTED_MD):
            assert "deleted within about a minute" in notice
            assert "never touches disk" not in notice
        assert "127.0.0.1" in ui_text.PRIVACY_LOCAL_MD
        assert "https://huggingface.co/privacy" in ui_text.PRIVACY_HOSTED_MD


class TestPlaceholders:
    @pytest.mark.parametrize("key", _EXPECTED_KEYS)
    def test_placeholder_set_matches_the_documented_list(self, key: str) -> None:
        expected = _EXPECTED_PLACEHOLDERS.get(key, set())
        assert _placeholders(getattr(ui_text, key)) == expected

    @pytest.mark.parametrize("key", sorted(_EXPECTED_PLACEHOLDERS))
    def test_parameterized_string_formats_cleanly(self, key: str) -> None:
        template: str = getattr(ui_text, key)
        fills = {name: _SAMPLE_FILL[name] for name in _EXPECTED_PLACEHOLDERS[key]}
        formatted = template.format(**fills)
        for value in fills.values():
            assert value in formatted


class TestRegister:
    def test_no_exclamation_marks_anywhere(self) -> None:
        offending = [key for key, text in _all_strings().items() if "!" in text]
        assert not offending, offending

    def test_no_emoji_anywhere(self) -> None:
        offending = [
            key for key, text in _all_strings().items() if _EMOJI_PATTERN.search(text)
        ]
        assert not offending, offending

    def test_no_confidence_or_certainty_language(self) -> None:
        """House rule in UI_COPY.md; 'uncertainty' (interval semantics) is allowed."""
        pattern = re.compile(r"\b(confidence|certainty)\b", re.IGNORECASE)
        offending = [key for key, text in _all_strings().items() if pattern.search(text)]
        assert not offending, offending

    def test_no_diagnosis_verb_addressed_to_the_user(self) -> None:
        pattern = re.compile(r"\byour (result|diagnosis)\b", re.IGNORECASE)
        offending = [key for key, text in _all_strings().items() if pattern.search(text)]
        assert not offending, offending


class TestInstallCommandRule:
    """The PyPI name 'retguard' is third-party: installs must use the GitHub URL."""

    def test_missing_gradio_hint_points_at_the_github_tag(self) -> None:
        assert (
            "git+https://github.com/anchor-neuro/retguard@v1.1.1"
            in ui_text.SERVE_MISSING_GRADIO
        )
        assert "[ui]" in ui_text.SERVE_MISSING_GRADIO

    def test_no_bare_pypi_install_in_package_or_docs(self) -> None:
        scanned = [
            _REPO_ROOT / "README.md",
            _REPO_ROOT / "CHANGELOG.md",
            _REPO_ROOT / "CONTRIBUTING.md",
            *sorted((_REPO_ROOT / "docs").glob("*.md")),
            *sorted((_REPO_ROOT / "retguard").glob("*.py")),
            *sorted((_REPO_ROOT / "examples").glob("*")),
        ]
        offending = []
        for path in scanned:
            if not path.is_file():
                continue
            # Quote stripping and whitespace collapse rejoin string literals
            # that source formatting wraps across lines.
            normalized = re.sub(
                r"\s+", " ", path.read_text(encoding="utf-8").replace('"', "")
            )
            for match in re.finditer(r"pip install", normalized):
                window = normalized[match.start() : match.start() + 200]
                if "git+https://github.com/anchor-neuro/retguard" not in window:
                    offending.append(f"{path.name}: {window[:80]}")
        assert not offending, offending
