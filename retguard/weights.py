# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Release-asset registry, download, and SHA-256 integrity verification.

Weights are distributed via GitHub Releases, never via the git tree. The
SHA-256 digests in ``WEIGHTS`` are computed from the final shipped bytes
during the release build; verification fails closed while any digest is
unset, so an unverified artifact can never be silently accepted.
"""

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from retguard.constants import MODULES

RELEASE_TAG = "v1.0.0"
_RELEASE_URL_ROOT = (
    f"https://github.com/anchor-neuro/retguard/releases/download/{RELEASE_TAG}"
)
_DOWNLOAD_TIMEOUT_SECONDS = 60
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class WeightSpec:
    """One module's release asset and its integrity data.

    Attributes:
        module: Module identifier from ``MODULES``.
        asset_name: Release zip filename.
        url: Full release-asset download URL.
        sha256_zip: SHA-256 of the zip; empty until the release build injects it.
        files: Extracted member filename to SHA-256; empty digests fail closed.
        size_bytes: Zip size in bytes; 0 until the release build injects it.
    """

    module: str
    asset_name: str
    url: str
    sha256_zip: str
    files: dict[str, str]
    size_bytes: int


# SHA-256 digests of the v1.0.0 release assets, computed from the final
# shipped bytes after ONNX license-metadata embedding; verification fails
# closed if any digest is empty.
WEIGHTS: dict[str, WeightSpec] = {
    "dr": WeightSpec(
        module="dr",
        asset_name="retguard-dr-v1.0.0.zip",
        url=f"{_RELEASE_URL_ROOT}/retguard-dr-v1.0.0.zip",
        sha256_zip="663c7f8bbc3676df2fe353efa0f3913c757db8c35cf9bf97ab451a756792d2e0",
        files={
            "retguard_dr_v1.0.0.onnx": "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b",
            "ood_gate_dr_v1.0.0.npz": "cb99b4d375ecc384ce6fbeff4f9b94c42613964b32b7a96cef8b3e1227dbf5b2",
        },
        size_bytes=204707438,
    ),
    "glaucoma": WeightSpec(
        module="glaucoma",
        asset_name="retguard-glaucoma-v1.0.0.zip",
        url=f"{_RELEASE_URL_ROOT}/retguard-glaucoma-v1.0.0.zip",
        sha256_zip="3b257b3e2c05469673a0d6ae7b633ac65a5a16c2030f779f546f4586eb166669",
        files={
            "retguard_glaucoma_v1.0.0.onnx": "b5686229ecc9050caddf61a66686e77e3e5cb4085425cca881078acb586cbb7b",
            "ood_gate_glaucoma_v1.0.0.npz": "5d07b5fbd8c0a77b8319e09f7e2b26b95ce1cc5b006705aec4864dd119ebde8d",
            "venn_abers_glaucoma_v1.0.0.npz": "8641e854024ae1ff2bdb8ecfdc6f6edb1a7dbf79674c9252c23f490437c5c405",
        },
        size_bytes=204819003,
    ),
    "oct": WeightSpec(
        module="oct",
        asset_name="retguard-oct-v1.0.0.zip",
        url=f"{_RELEASE_URL_ROOT}/retguard-oct-v1.0.0.zip",
        sha256_zip="5b8f659df824174217031480ecd2ac767bd2e30fdbe928070d97575739b18bd1",
        files={
            "retguard_oct_v1.0.0.onnx": "5ebd2e814a718edca922c26f5bdce380c6b38506f923103a8cb3362b67fb75f3",
            "ood_gate_oct_v1.0.0.npz": "68fc53ad187bdb14f2dae53fd41ad25bc563fc7c8d7cad6517d065578fe87532",
            "retguard_oct_v1.0.0.onnx.data": "88d6c4d6803d3201b182eeb528b9ab08f2641de2b1d1ef672c807f9ecda5243c",
            "venn_abers_oct_v1.0.0.npz": "176c74b62e8b100d186249dd6f63dac877ca13385c6d38254be774f8cd05a5a4",
        },
        size_bytes=204678963,
    ),
}


def default_weights_dir() -> Path:
    """Return the weights directory: ``$RETGUARD_WEIGHTS_DIR`` or ``~/.retguard``."""
    env_dir = os.environ.get("RETGUARD_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".retguard"


def artifact_paths(module: str, weights_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve a module's extracted artifact paths.

    Args:
        module: One of ``MODULES``.
        weights_dir: Directory holding extracted release members. Defaults to
            ``default_weights_dir()``.

    Returns:
        Mapping with keys ``"onnx"``, ``"ood"``, and (glaucoma, oct)
        ``"venn_abers"``; oct additionally has ``"onnx_data"``.

    Raises:
        ValueError: If ``module`` is not a known module identifier.
    """
    if module not in MODULES:
        raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
    base = Path(weights_dir) if weights_dir is not None else default_weights_dir()
    paths = {
        "onnx": base / f"retguard_{module}_{RELEASE_TAG}.onnx",
        "ood": base / f"ood_gate_{module}_{RELEASE_TAG}.npz",
    }
    if module == "oct":
        paths["onnx_data"] = base / f"retguard_oct_{RELEASE_TAG}.onnx.data"
    if module in ("glaucoma", "oct"):
        paths["venn_abers"] = base / f"venn_abers_{module}_{RELEASE_TAG}.npz"
    return paths


def download(
    module: str, dest_dir: Path | None = None, force: bool = False
) -> Path:
    """Download, verify, and extract one module's release asset.

    Args:
        module: One of ``MODULES``.
        dest_dir: Extraction directory. Defaults to ``default_weights_dir()``.
        force: Re-download even if all members already exist and verify.

    Returns:
        The directory the members were extracted into.

    Raises:
        ValueError: If ``module`` is not a known module identifier.
        RuntimeError: If the checksum registry is unpopulated or any SHA-256
            digest does not match the downloaded bytes.
    """
    if module not in MODULES:
        raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
    spec = WEIGHTS[module]
    _require_populated(spec)
    dest_dir = dest_dir if dest_dir is not None else default_weights_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((dest_dir / name).is_file() for name in spec.files):
        verify(module, dest_dir)
        return dest_dir

    zip_path = dest_dir / spec.asset_name
    with urllib.request.urlopen(
        spec.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
    ) as response, open(zip_path, "wb") as zip_file:
        while True:
            chunk = response.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            zip_file.write(chunk)

    observed_zip_digest = _sha256_file(zip_path)
    if observed_zip_digest != spec.sha256_zip:
        zip_path.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for {spec.asset_name}: expected {spec.sha256_zip}, "
            f"got {observed_zip_digest}. Re-download with --force."
        )
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    zip_path.unlink()
    verify(module, dest_dir)
    return dest_dir


def verify(module: str, weights_dir: Path | None = None) -> None:
    """Re-hash a module's extracted members against the registry.

    Args:
        module: One of ``MODULES``.
        weights_dir: Directory holding the extracted members. Defaults to
            ``default_weights_dir()``.

    Raises:
        ValueError: If ``module`` is not a known module identifier.
        FileNotFoundError: If an expected member file is absent.
        RuntimeError: If the checksum registry is unpopulated or any member's
            SHA-256 does not match.
    """
    if module not in MODULES:
        raise ValueError(f"module must be one of {MODULES}, got {module!r}.")
    spec = WEIGHTS[module]
    _require_populated(spec)
    base = weights_dir if weights_dir is not None else default_weights_dir()
    for member_name, expected_digest in spec.files.items():
        member_path = base / member_name
        if not member_path.is_file():
            raise FileNotFoundError(
                f"Weights member not found at {member_path}. "
                f"Run 'retguard download --module {module}' (see README)."
            )
        observed_digest = _sha256_file(member_path)
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"SHA-256 mismatch for {member_path.name}: expected "
                f"{expected_digest}, got {observed_digest}. Re-download with --force."
            )


def _require_populated(spec: WeightSpec) -> None:
    """Fail closed while any digest in the registry is unset."""
    if not spec.sha256_zip or any(not digest for digest in spec.files.values()):
        raise RuntimeError(
            f"Checksum registry for module {spec.module!r} is not populated; "
            "refusing to download or verify without SHA-256 digests. "
            "This build predates the release; use a released version."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as binary_file:
        while True:
            chunk = binary_file.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
