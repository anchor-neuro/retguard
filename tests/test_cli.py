# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""CLI argument parsing and the batch-loop error contract, with stubbed backends."""

import json
from pathlib import Path

import pytest

from retguard import cli
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
