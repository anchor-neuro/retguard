# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""CLI argument parsing and the batch-loop error contract, with stubbed backends."""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

from retguard import cli, ui_text
from retguard.predictor import PredictionResult


def _result(probability: float = 0.9) -> PredictionResult:
    return PredictionResult(
        module="oct",
        probability=probability,
        decision=probability >= 0.7,
        threshold=0.7,
        class_probabilities={"Normal": 0.05, "AMD": probability, "DME": 0.05},
        venn_abers_interval=(probability - 0.01, probability + 0.01),
        ood_score=12.0,
        ood_flagged=False,
    )


class _StubPredictor:
    def __init__(self, failing_inputs: set[str]) -> None:
        self.failing_inputs = failing_inputs
        self.seen: list[str] = []

    def predict(self, image_path: str) -> PredictionResult:
        self.seen.append(image_path)
        if image_path in self.failing_inputs:
            raise ValueError(f"Cannot decode {image_path} as an image.")
        return _result()


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["retguard", *argv])
    return cli.main()


def test_no_subcommand_is_an_argparse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, [])
    assert excinfo.value.code == 2


def test_unknown_module_is_an_argparse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, ["download", "--module", "fundus"])
    assert excinfo.value.code == 2


def test_predict_requires_module_and_input(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, ["predict", "--module", "oct"])
    assert excinfo.value.code == 2


def test_download_all_iterates_every_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[tuple[str, Path | None, bool]] = []

    def stub_download(module: str, dest_dir: Path | None = None, force: bool = False) -> Path:
        calls.append((module, dest_dir, force))
        return tmp_path

    monkeypatch.setattr(cli, "download", stub_download)
    assert _run(monkeypatch, ["download"]) == 0
    assert [call[0] for call in calls] == ["dr", "glaucoma", "oct"]
    assert all(not call[2] for call in calls)
    assert capsys.readouterr().out.count("downloaded and verified") == 3


def test_download_single_module_with_force_and_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path | None, bool]] = []

    def stub_download(module: str, dest_dir: Path | None = None, force: bool = False) -> Path:
        calls.append((module, dest_dir, force))
        return tmp_path

    monkeypatch.setattr(cli, "download", stub_download)
    exit_code = _run(
        monkeypatch,
        ["download", "--module", "dr", "--weights-dir", str(tmp_path), "--force"],
    )
    assert exit_code == 0
    assert calls == [("dr", tmp_path, True)]


def test_verify_single_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli, "verify", lambda module, weights_dir=None: calls.append(module)
    )
    assert _run(monkeypatch, ["verify", "--module", "glaucoma"]) == 0
    assert calls == ["glaucoma"]
    assert "all SHA-256 digests match" in capsys.readouterr().out


def test_predict_prints_one_line_per_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = _StubPredictor(failing_inputs=set())
    monkeypatch.setattr(cli, "load", lambda module, weights_dir=None: stub)
    exit_code = _run(monkeypatch, ["predict", "--module", "oct", "a.jpeg", "b.jpeg"])
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert "probability=0.900000" in lines[0]
    assert "decision=True" in lines[0]
    assert "ood_flagged=False" in lines[0]


def test_predict_json_output_parses_with_complete_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = _StubPredictor(failing_inputs=set())
    monkeypatch.setattr(cli, "load", lambda module, weights_dir=None: stub)
    exit_code = _run(monkeypatch, ["predict", "--module", "oct", "a.jpeg", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["input"] == "a.jpeg"
    for field in (
        "module",
        "probability",
        "decision",
        "threshold",
        "class_probabilities",
        "venn_abers_interval",
        "ood_score",
        "ood_flagged",
    ):
        assert field in entry


def test_serve_flag_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(
        cli, "_run_serve", lambda arguments: seen.append(arguments) or 0
    )
    assert _run(monkeypatch, ["serve"]) == 0
    arguments = seen[0]
    assert arguments.port is None
    assert arguments.host == "127.0.0.1"
    assert arguments.no_browser is False
    assert arguments.weights_dir is None
    assert arguments.examples_dir is None


def test_serve_flags_are_all_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(
        cli, "_run_serve", lambda arguments: seen.append(arguments) or 0
    )
    exit_code = _run(
        monkeypatch,
        [
            "serve",
            "--port",
            "7999",
            "--host",
            "0.0.0.0",
            "--no-browser",
            "--weights-dir",
            str(tmp_path),
            "--examples-dir",
            str(tmp_path),
        ],
    )
    assert exit_code == 0
    arguments = seen[0]
    assert arguments.port == 7999
    assert arguments.host == "0.0.0.0"
    assert arguments.no_browser is True
    assert arguments.weights_dir == tmp_path
    assert arguments.examples_dir == tmp_path


class _StubDemo:
    local_url = "http://127.0.0.1:7860/"

    def block_thread(self) -> None:
        return None


def _install_ui_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, str | None]]:
    """Replace retguard.ui with a recorder so serve runs without gradio."""
    launch_kwargs: dict[str, object] = {}
    env_at_build: dict[str, str | None] = {}

    def stub_build_demo(
        weights_dir: Path | None = None,
        examples_dir: Path | None = None,
        hosted: bool = False,
    ) -> _StubDemo:
        env_at_build["analytics"] = os.environ.get("GRADIO_ANALYTICS_ENABLED")
        env_at_build["temp_dir"] = os.environ.get("GRADIO_TEMP_DIR")
        return _StubDemo()

    def stub_launch_demo(demo: _StubDemo, **kwargs: object) -> None:
        launch_kwargs.update(kwargs)

    stub_module = types.SimpleNamespace(
        build_demo=stub_build_demo, launch_demo=stub_launch_demo
    )
    monkeypatch.setitem(sys.modules, "retguard.ui", stub_module)
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)
    return launch_kwargs, env_at_build


def test_serve_sets_zero_persistence_env_before_the_ui_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launch_kwargs, env_at_build = _install_ui_stub(monkeypatch)
    assert _run(monkeypatch, ["serve", "--no-browser"]) == 0
    assert env_at_build["analytics"] == "False"
    temp_dir = env_at_build["temp_dir"]
    assert temp_dir is not None
    assert "retguard-serve-" in Path(temp_dir).name
    # The dedicated temp dir exists for the session and is removed whole at
    # shutdown - gradio's own sweep leaves cached example copies behind.
    assert not Path(temp_dir).exists()
    assert "share" not in launch_kwargs
    capsys.readouterr()


def test_serve_delegates_port_fallback_to_gradio_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """server_port=None makes gradio walk up from 7860 to the next free port."""
    launch_kwargs, env_at_build = _install_ui_stub(monkeypatch)
    assert _run(monkeypatch, ["serve"]) == 0
    assert launch_kwargs["server_port"] is None
    assert launch_kwargs["server_name"] == "127.0.0.1"
    assert launch_kwargs["inbrowser"] is True
    assert launch_kwargs["max_file_size"] == "30mb"
    assert not Path(str(env_at_build["temp_dir"])).exists()
    capsys.readouterr()


def test_serve_passes_an_explicit_port_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launch_kwargs, env_at_build = _install_ui_stub(monkeypatch)
    assert _run(monkeypatch, ["serve", "--port", "7999", "--no-browser"]) == 0
    assert launch_kwargs["server_port"] == 7999
    assert launch_kwargs["inbrowser"] is False
    assert not Path(str(env_at_build["temp_dir"])).exists()
    capsys.readouterr()


def test_serve_allowlists_the_examples_dir_for_gradio_file_serving(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Without allowed_paths an example click 500s when the dir is outside cwd."""
    launch_kwargs, env_at_build = _install_ui_stub(monkeypatch)
    exit_code = _run(
        monkeypatch,
        ["serve", "--no-browser", "--examples-dir", str(tmp_path)],
    )
    assert exit_code == 0
    assert launch_kwargs["allowed_paths"] == [str(tmp_path)]
    assert not Path(str(env_at_build["temp_dir"])).exists()
    capsys.readouterr()


def test_serve_without_an_examples_dir_allowlists_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launch_kwargs, env_at_build = _install_ui_stub(monkeypatch)
    assert _run(monkeypatch, ["serve", "--no-browser"]) == 0
    assert launch_kwargs["allowed_paths"] == []
    assert not Path(str(env_at_build["temp_dir"])).exists()
    capsys.readouterr()


def test_serve_prints_exactly_the_four_terminal_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, env_at_build = _install_ui_stub(monkeypatch)
    assert _run(monkeypatch, ["serve", "--no-browser"]) == 0
    assert not Path(str(env_at_build["temp_dir"])).exists()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[0] == ui_text.SERVE_TERMINAL_BANNER
    assert lines[1].startswith("Serving at http://127.0.0.1:7860/ (models loaded in ")
    assert lines[1].endswith(" s).")
    assert lines[2] == ui_text.SERVE_TERMINAL_OFFLINE
    assert lines[3] == ui_text.SERVE_TERMINAL_STOP
    assert all("!" not in line for line in lines)


def test_serve_without_gradio_prints_the_install_hint_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "retguard.ui", None)
    assert _run(monkeypatch, ["serve"]) == 1
    assert ui_text.SERVE_MISSING_GRADIO in capsys.readouterr().err


def test_predict_batch_continues_past_a_bad_input_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = _StubPredictor(failing_inputs={"bad.jpeg"})
    monkeypatch.setattr(cli, "load", lambda module, weights_dir=None: stub)
    exit_code = _run(
        monkeypatch, ["predict", "--module", "dr", "good.jpeg", "bad.jpeg", "also_good.jpeg"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert stub.seen == ["good.jpeg", "bad.jpeg", "also_good.jpeg"]
    assert "bad.jpeg: ValueError" in captured.err
    assert len(captured.out.strip().splitlines()) == 2
