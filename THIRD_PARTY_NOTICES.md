# Third-Party Notices

Runtime dependencies of the `retguard` package, with the exact versions pinned
in `requirements-lock.txt` and the license reported by each installed package's
metadata.

| Package | Version pin | License | Role |
|---|---|---|---|
| onnxruntime | 1.29.0 | MIT | ONNX model inference (CPUExecutionProvider) |
| numpy | 2.5.2 | BSD-3-Clause (metadata expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) | Tensor math, calibration, and OOD-gate linear algebra |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (package metadata; bundles OpenCV, Apache-2.0) | Image decoding, Lanczos4 resize, Gaussian blur, CLAHE |
| onnx | 1.22.0 | Apache-2.0 | In-memory graph augmentation only: exposes the 1,280-d post-GAP feature tensor consumed by the OOD gate |

All four licenses are permissive and impose no restriction on this repository's
research-use terms.
