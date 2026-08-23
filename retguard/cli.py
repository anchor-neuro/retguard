# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Sameh Aboelmaaty / Anchor Neuro. See LICENSE.md and COMMERCIAL-LICENSE.md.
"""Console entry point: ``retguard download | verify | predict``."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from retguard.constants import MODULES
from retguard.predictor import load
from retguard.weights import download, verify


def main() -> int:
    """Parse arguments and dispatch to a subcommand.

    Returns:
        Process exit code: 0 on success, 1 if any input failed.
    """
    parser = argparse.ArgumentParser(
        prog="retguard",
        description=(
            "RETGUARD retinal screening inference (research use only; "
            "not a medical device)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download and verify release weights."
    )
    download_parser.add_argument(
        "--module", choices=(*MODULES, "all"), default="all"
    )
    download_parser.add_argument("--weights-dir", type=Path, default=None)
    download_parser.add_argument("--force", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="Re-verify downloaded weights against SHA-256 digests."
    )
    verify_parser.add_argument("--module", choices=(*MODULES, "all"), default="all")
    verify_parser.add_argument("--weights-dir", type=Path, default=None)

    predict_parser = subparsers.add_parser(
        "predict", help="Run one module on one or more images."
    )
    predict_parser.add_argument("--module", choices=MODULES, required=True)
    predict_parser.add_argument("inputs", nargs="+", metavar="INPUT")
    predict_parser.add_argument("--weights-dir", type=Path, default=None)
    predict_parser.add_argument("--json", action="store_true", dest="as_json")

    arguments = parser.parse_args()
    if arguments.command == "download":
        return _run_download(arguments)
    if arguments.command == "verify":
        return _run_verify(arguments)
    return _run_predict(arguments)


def _selected_modules(module_argument: str) -> tuple[str, ...]:
    return MODULES if module_argument == "all" else (module_argument,)


def _run_download(arguments: argparse.Namespace) -> int:
    for module in _selected_modules(arguments.module):
        extracted_dir = download(
            module, dest_dir=arguments.weights_dir, force=arguments.force
        )
        print(f"{module}: downloaded and verified in {extracted_dir}")
    return 0


def _run_verify(arguments: argparse.Namespace) -> int:
    for module in _selected_modules(arguments.module):
        verify(module, weights_dir=arguments.weights_dir)
        print(f"{module}: all SHA-256 digests match")
    return 0


def _run_predict(arguments: argparse.Namespace) -> int:
    predictor = load(arguments.module, weights_dir=arguments.weights_dir)
    results = []
    any_failed = False
    for input_path in arguments.inputs:
        # The repo's single permitted broad handler: a bad file must not abort
        # the remaining batch (Whisper CLI pattern).
        try:
            result = predictor.predict(input_path)
        except Exception as error:  # noqa: BLE001
            any_failed = True
            print(
                f"{input_path}: {type(error).__name__}: {error}", file=sys.stderr
            )
            continue
        results.append((input_path, result))

    if arguments.as_json:
        print(
            json.dumps(
                [
                    {"input": input_path, **dataclasses.asdict(result)}
                    for input_path, result in results
                ],
                indent=2,
            )
        )
    else:
        for input_path, result in results:
            print(
                f"{input_path}  probability={result.probability:.6f}  "
                f"decision={result.decision}  ood_flagged={result.ood_flagged}"
            )
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
