# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Weights registry integrity and fail-closed verification behavior."""

import dataclasses
import hashlib
from pathlib import Path

import pytest

from retguard.constants import MODULES
from retguard.weights import (
    RELEASE_TAG,
    WEIGHTS,
    WeightSpec,
    artifact_paths,
    default_weights_dir,
    verify,
)


def test_registry_covers_every_module() -> None:
    assert tuple(WEIGHTS) == MODULES


def test_registry_has_no_empty_hash_or_url() -> None:
    for spec in WEIGHTS.values():
        assert spec.sha256_zip, f"{spec.module}: empty zip digest"
        assert spec.url.startswith("https://"), f"{spec.module}: bad url {spec.url}"
        assert spec.size_bytes > 0, f"{spec.module}: zero size"
        assert spec.files, f"{spec.module}: no member files"
        for member_name, digest in spec.files.items():
            assert digest, f"{spec.module}/{member_name}: empty member digest"
            assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_registry_asset_names_follow_the_release_convention() -> None:
    for module, spec in WEIGHTS.items():
        assert spec.asset_name == f"retguard-{module}-{RELEASE_TAG}.zip"
        assert spec.url.endswith(spec.asset_name)
        assert f"retguard_{module}_{RELEASE_TAG}.onnx" in spec.files
        assert f"ood_gate_{module}_{RELEASE_TAG}.npz" in spec.files
    assert f"venn_abers_glaucoma_{RELEASE_TAG}.npz" in WEIGHTS["glaucoma"].files
    assert f"venn_abers_oct_{RELEASE_TAG}.npz" in WEIGHTS["oct"].files
    assert f"retguard_oct_{RELEASE_TAG}.onnx.data" in WEIGHTS["oct"].files


def test_default_weights_dir_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RETGUARD_WEIGHTS_DIR", str(tmp_path))
    assert default_weights_dir() == tmp_path
    monkeypatch.delenv("RETGUARD_WEIGHTS_DIR")
    assert default_weights_dir() == Path.home() / ".retguard"


def test_artifact_paths_keys_per_module() -> None:
    assert set(artifact_paths("dr", "w")) == {"onnx", "ood"}
    assert set(artifact_paths("glaucoma", "w")) == {"onnx", "ood", "venn_abers"}
    assert set(artifact_paths("oct", "w")) == {"onnx", "onnx_data", "ood", "venn_abers"}


def test_artifact_paths_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="module must be one of"):
        artifact_paths("fundus")


def test_verify_raises_on_a_corrupted_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    good_bytes = b"artifact bytes"
    corrupt_member = f"ood_gate_dr_{RELEASE_TAG}.npz"
    spec = WEIGHTS["dr"]
    populated_files = {}
    for member_name in spec.files:
        member_path = tmp_path / member_name
        member_path.write_bytes(good_bytes)
        populated_files[member_name] = hashlib.sha256(good_bytes).hexdigest()
    monkeypatch.setitem(
        WEIGHTS,
        "dr",
        dataclasses.replace(spec, sha256_zip="0" * 64, files=populated_files),
    )
    verify("dr", tmp_path)
    (tmp_path / corrupt_member).write_bytes(b"tampered bytes")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify("dr", tmp_path)


def test_verify_raises_on_a_missing_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = WEIGHTS["dr"]
    monkeypatch.setitem(
        WEIGHTS,
        "dr",
        dataclasses.replace(
            spec,
            sha256_zip="0" * 64,
            files={name: "0" * 64 for name in spec.files},
        ),
    )
    with pytest.raises(FileNotFoundError, match="retguard download"):
        verify("dr", tmp_path)


def test_unpopulated_registry_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = WEIGHTS["dr"]
    empty_spec = WeightSpec(
        module=spec.module,
        asset_name=spec.asset_name,
        url=spec.url,
        sha256_zip="",
        files={name: "" for name in spec.files},
        size_bytes=0,
    )
    monkeypatch.setitem(WEIGHTS, "dr", empty_spec)
    with pytest.raises(RuntimeError, match="not populated"):
        verify("dr", tmp_path)


def test_verify_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="module must be one of"):
        verify("fundus")
