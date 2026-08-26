# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Headless smoke against the real models: ack flow, full results, OOD, cleanup.

Launches the Blocks app in-process and drives it with ``gradio_client``.
The tests in this module are order-dependent by design: acknowledgment is a
session-level state transition, and the temp-dir sweep is checked last.
"""

import json
import os
import re
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import make_synthetic_fundus, make_synthetic_oct

from retguard import ui_text
from retguard.constants import MODULES
from retguard.predictor import load

pytestmark = pytest.mark.requires_weights

gradio_client = pytest.importorskip("gradio_client")

_UI_CLI_PARITY_TOLERANCE = 1e-6
# delete_cache=(60, 60) purges files older than 60 s every 60 s; polling past
# two full sweep periods bounds the wait (V11_BLUEPRINT section 6.2).
_CLEANUP_POLL_SECONDS = 180


@pytest.fixture(scope="module")
def gradio_temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    temp_dir = tmp_path_factory.mktemp("retguard-smoke-cache")
    previous = os.environ.get("GRADIO_TEMP_DIR")
    os.environ["GRADIO_TEMP_DIR"] = str(temp_dir)
    yield temp_dir
    if previous is None:
        os.environ.pop("GRADIO_TEMP_DIR", None)
    else:
        os.environ["GRADIO_TEMP_DIR"] = previous


@pytest.fixture(scope="module")
def examples_dir() -> Path | None:
    """Directory with the eight curated example images, or None.

    Set RETGUARD_EXAMPLES_DIR to the staged Space examples/ tree to run the
    bundled-example smoke; without it the gallery is simply omitted, exactly
    as ``retguard serve`` without --examples-dir.
    """
    value = os.environ.get("RETGUARD_EXAMPLES_DIR")
    return Path(value) if value else None


@pytest.fixture(scope="module")
def served_demo(
    weights_dir: Path, gradio_temp_dir: Path, examples_dir: Path | None
) -> Iterator[object]:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    from retguard.ui import build_demo, launch_demo

    demo = build_demo(weights_dir=weights_dir, examples_dir=examples_dir)
    # allowed_paths mirrors cli._run_serve: without it every example click
    # 500s when the examples dir sits outside the process cwd.
    launch_demo(
        demo,
        server_name="127.0.0.1",
        prevent_thread_lock=True,
        quiet=True,
        inbrowser=False,
        show_error=True,
        allowed_paths=[str(examples_dir)] if examples_dir is not None else [],
    )
    yield demo
    demo.close()


@pytest.fixture()
def bundled_examples(examples_dir: Path | None) -> Path:
    if examples_dir is None:
        pytest.skip("set RETGUARD_EXAMPLES_DIR to run the bundled-example smoke")
    return examples_dir


@pytest.fixture(scope="module")
def client(served_demo: object) -> object:
    return gradio_client.Client(served_demo.local_url)


@pytest.fixture(scope="module")
def image_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    image_dir = tmp_path_factory.mktemp("retguard-smoke-images")
    rng = np.random.default_rng(42)
    images = {
        "dr": make_synthetic_fundus(seed=3),
        "glaucoma": make_synthetic_fundus(seed=4),
        "oct": make_synthetic_oct(seed=5),
        "noise": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
    }
    paths = {}
    for name, image in images.items():
        path = image_dir / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths[name] = path
    return paths


def _analyze(client: object, image_path: Path, module: str) -> tuple[object, ...]:
    response = client.predict(  # type: ignore[attr-defined]
        gradio_client.handle_file(str(image_path)),
        ui_text.TTA_FAITHFUL_LABEL,
        api_name=f"/analyze_{module}",
    )
    return tuple(response) if isinstance(response, (tuple, list)) else (response,)


def _find_fragment(response: tuple[object, ...], marker: str) -> str:
    for item in response:
        if isinstance(item, str) and marker in item:
            return item
    raise AssertionError(f"no response fragment contains {marker!r}: {response!r}")


def _details_probability(response: tuple[object, ...]) -> float:
    details = _find_fragment(response, "```json")
    match = re.search(r"```json\n(.*?)\n```", details, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    return float(payload["probability"])


def test_analyze_before_acknowledgment_raises(
    client: object, image_paths: dict[str, Path]
) -> None:
    with pytest.raises(Exception, match="I understand"):
        _analyze(client, image_paths["dr"], "dr")


def test_acknowledge_endpoint_completes(client: object) -> None:
    client.predict(api_name="/acknowledge")  # type: ignore[attr-defined]


@pytest.mark.parametrize("module", MODULES)
def test_full_result_renders_and_matches_the_predictor(
    client: object,
    image_paths: dict[str, Path],
    weights_dir: Path,
    module: str,
) -> None:
    response = _analyze(client, image_paths[module], module)
    panel = _find_fragment(response, "retguard-card")
    assert "DISTRIBUTION" in panel
    assert "Model output:" in panel
    provenance = _find_fragment(response, "retguard-provenance")
    assert "retguard 1.1.1" in provenance
    expected_views = 2 if module == "oct" else 8
    assert f"{expected_views}-view" in provenance
    if module == "oct":
        assert "347 of 354" in panel

    ui_probability = _details_probability(response)
    reference = load(module, weights_dir=weights_dir).predict(image_paths[module])
    assert abs(ui_probability - reference.probability) < _UI_CLI_PARITY_TOLERANCE


@pytest.mark.parametrize("module", MODULES)
def test_non_retinal_image_is_flagged_out_of_distribution(
    client: object, image_paths: dict[str, Path], module: str
) -> None:
    response = _analyze(client, image_paths["noise"], module)
    panel = _find_fragment(response, "retguard-card")
    assert "OUT OF DISTRIBUTION" in panel
    assert "retguard-unreliable" in panel
    details = _find_fragment(response, "```json")
    assert '"ood_flagged": true' in details


@pytest.mark.parametrize("module", MODULES)
def test_every_bundled_example_analyzes_on_its_tab(
    client: object, bundled_examples: Path, module: str
) -> None:
    """KG-2 with the shipped gallery: every curated example produces a full
    result on its tab, and not_retinal.jpg trips the OOD gate everywhere."""
    from retguard.ui import _EXAMPLE_FILES

    for filename, _ in _EXAMPLE_FILES[module]:
        example_path = bundled_examples / filename
        assert example_path.is_file(), example_path
        response = _analyze(client, example_path, module)
        panel = _find_fragment(response, "retguard-card")
        assert "Model output:" in panel
        if filename == "not_retinal.jpg":
            assert "OUT OF DISTRIBUTION" in panel
            details = _find_fragment(response, "```json")
            assert '"ood_flagged": true' in details
        else:
            assert "IN DISTRIBUTION" in panel


def test_example_files_are_served_from_the_allowed_examples_dir(
    served_demo: object, client: object, bundled_examples: Path
) -> None:
    """Regression for the example-click 500: gradio must be allowed to serve
    files straight from the examples dir (launch-level allowed_paths)."""
    example_path = (bundled_examples / "not_retinal.jpg").resolve()
    file_url = (
        served_demo.local_url  # type: ignore[attr-defined]
        + "gradio_api/file="
        + str(example_path)
    )
    with urllib.request.urlopen(file_url, timeout=10) as response:
        assert response.status == 200
        assert response.read(4) != b""


def test_served_page_carries_banner_and_disclaimer(served_demo: object) -> None:
    config_url = served_demo.local_url + "config"
    with urllib.request.urlopen(config_url, timeout=10) as response:
        config_text = response.read().decode("utf-8")
    assert "research demonstration only" in config_text
    assert ui_text.RESULT_DISCLAIMER_MD in config_text
    assert ui_text.ACK_BUTTON in config_text


def test_temp_dir_holds_no_stale_files_after_the_session(
    served_demo: object, gradio_temp_dir: Path, examples_dir: Path | None
) -> None:
    # Bundled example images are cached with keep_in_cache for the server's
    # lifetime by design (public licensed images, never user data); every
    # user-derived file must still be swept while the server runs, and after
    # close nothing at all may remain.
    example_filenames = (
        {path.name for path in examples_dir.iterdir()}
        if examples_dir is not None
        else set()
    )
    deadline = time.monotonic() + _CLEANUP_POLL_SECONDS
    remaining: list[Path] = []
    while time.monotonic() < deadline:
        remaining = [
            path
            for path in gradio_temp_dir.rglob("*")
            if path.is_file() and path.name not in example_filenames
        ]
        if not remaining:
            break
        time.sleep(5)
    assert not remaining, remaining
    assert not Path(".gradio/flagged").exists()
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False"
    served_demo.close()  # type: ignore[attr-defined]
    # Shutdown leaves the keep_in_cache copies of the app's own bundled
    # examples (cli._run_serve removes its whole dedicated temp dir instead);
    # what may never survive is anything user-derived.
    post_close = [
        path
        for path in gradio_temp_dir.rglob("*")
        if path.is_file() and path.name not in example_filenames
    ]
    assert not post_close, post_close
