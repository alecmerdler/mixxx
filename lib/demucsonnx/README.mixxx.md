# demucsonnx (vendored)

Source: https://github.com/sevagh/demucs.onnx
Vendored subset: `src/` (inference library only, excludes `src_cli/` and tests)
Commit: 81fa192e6fcc88e35e887f6e6ccce91227f4e6f5
License: MIT (see LICENSE)

Only the inference library is vendored here. Audio I/O, which upstream
implements with libnyquist in `src_cli/`, is handled by Mixxx's own
decoder/encoder infrastructure instead (see
`src/track/stemseparation/stemseparator.cpp`).

Still depends on Eigen (for STFT/iSTFT buffers and tensor math around the
ONNX Runtime calls, which only run the core neural network); the actual
neural network inference is delegated to ONNX Runtime
(`onnxruntime_cxx_api.h`) instead of being hand-written, which is why this
replaced `lib/demucscpp/` (CPU-only, hand-written Eigen layers) — ONNX
Runtime can use GPU/NPU execution providers where available and is
generally much faster even on CPU due to its own kernel optimizations.

Model weights are **not** vendored. See `STEMS.md` §7 for how to obtain
`.onnx` weights via upstream's `scripts/convert-pth-to-onnx.py`.
