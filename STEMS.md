> **AI-generated document.** This file was written autonomously by an AI coding
> assistant (Claude) during a pairing session, based on code it also wrote. It
> has **not** been reviewed, tested on real hardware/DJ gear, or approved by a
> human maintainer. Per `AGENTS.md`, nothing described here has been
> committed or submitted as a pull request — treat this as a design/status
> snapshot for a human contributor to review, test, and decide whether to
> pursue further.

# AI-Based Stem Separation — Status & Design

## 1. What this is

Mixxx already supports **pre-authored** STEM files (Native Instruments
`.stem.mp4`/`.stem.m4a`, up to 4 embedded stereo stems + a premix), with UI
for muting/soloing stems per deck (`StemInfo`, `WTrackStemMenu`,
`WStemLabel`, `EngineDeck` stem channels).

This feature adds **on-device AI source separation** of *arbitrary* tracks
(no pre-authored stems required), producing the same kind of STEM container
on demand, so that the existing STEM playback/UI pipeline can be reused
almost entirely unmodified.

High-level flow:

1. User loads a track and clicks "Separate Stems (AI)" (deck button or
   track context menu).
2. A background job decodes the track, runs 4-stem separation
   (drums/bass/other/vocals), and muxes the result + the original premix
   into a `.stem.m4a` container file in a per-track folder.
3. The `Track` object is updated with the path to that container
   (`stems_path`, persisted in the library DB).
4. Whenever that track is loaded for playback (now or in any future
   session), Mixxx transparently opens the generated STEM container instead
   of the original file, so stem mute/solo, colors, etc. "just work" via the
   pre-existing STEM engine code.

## 2. Status

**Prototype / unreviewed.** Implemented and verified to compile, link, and
run as part of a real Mixxx build (see §6), but:

- The inference library, and `StemSeparator`/`StemSeparatorCache` as
  compiled into `mixxx-lib`, have been exercised with real (official,
  converted) model weights and real ONNX Runtime inference, on both the
  CoreML and CPU execution providers (§6) — but the feature has not been
  run from the *running application* (UI triggers, container write,
  playback of the result) by a human.
- Not tested on a real DJ setup, per `AGENTS.md`'s requirement before any PR.
- Several follow-up items remain (see §7).

Nothing has been committed. All changes are in the working tree, gated
behind a new CMake option (`STEM_SEPARATION`, default `OFF`), so existing
Mixxx builds/behavior are unaffected unless a developer explicitly opts in.

## 3. Design decisions

### 3.1 Model & inference library: demucs.onnx (replacing demucs.cpp)

**Update:** this project originally used
[demucs.cpp](https://github.com/sevagh/demucs.cpp), a hand-written,
Eigen-only, CPU-bound reimplementation of htdemucs. It worked but was
measured as far too slow for practical use (tens of seconds per 30s audio
segment on a modern desktop CPU, with no path to using a GPU/NPU). It has
been replaced with
[**demucs.onnx**](https://github.com/sevagh/demucs.onnx), which keeps the
same htdemucs architecture and the same STFT/iSTFT pre/post-processing code
(adapted from demucs.cpp) but delegates the actual neural network forward
pass to **ONNX Runtime** instead of hand-written Eigen layers. This:

- Is dramatically faster even purely on CPU, since ONNX Runtime ships its
  own optimized/fused kernels (quantization, graph optimizations, SIMD)
  that a hand-written reference implementation doesn't attempt.
- Opens a path to GPU/NPU execution providers (CUDA, CoreML, DirectML)
  by simply changing `Ort::SessionOptions`, with no change to the vendored
  inference code. This is now implemented — see §3.6.
- Is still MIT-licensed, still has no PyTorch/Python dependency at runtime,
  and still outputs the same 4 sources (drums/bass/other/vocals) mapping
  1:1 onto the stem slots Mixxx's existing STEM UI already expects.

Trade-off accepted: this adds **ONNX Runtime** as a new, fairly large,
third-party runtime dependency (resolved via `find_package()`, e.g. via
Homebrew's `onnxruntime` formula or a vcpkg port, rather than vendored,
since it's a widely packaged binary library like Qt or FFmpeg). This is a
bigger ask for packagers/distros than demucs.cpp's Eigen3+OpenMP
dependency, but was judged worth it given how large the resulting speedup
is for what's meant to be a usable background feature, not a
ever-run-once prototype.

**Vendoring**: only the inference library (`src/` of upstream, ~900 lines
across 3 `.cpp` files) is vendored, under `lib/demucsonnx/`, matching
Mixxx's existing convention for small unpackaged dependencies (e.g.
`lib/qm-dsp`). Upstream's `src_cli/` (which uses `libnyquist` for file I/O)
is *not* vendored — Mixxx's own decoder (`SoundSourceProxy`) and a custom
FFmpeg muxer are used instead, same as before. `Eigen3` (still needed for
STFT/iSTFT buffers and tensor bookkeeping around the ONNX Runtime calls)
and `onnxruntime` are resolved via `find_package()`.

Model weights (ONNX format, converted from the official PyTorch checkpoint)
are **not** bundled or downloaded automatically — see §7.

### 3.2 Output container format: reuse the existing STEM engine

The alternative to this would have been inventing a new playback path (e.g.
4 separate files loaded into 4 "virtual decks"/slots). Instead, the
generated stems are muxed into a single container file with the **exact
stream layout `SoundSourceSTEM` already expects**: 1 premix stream + 4 stem
streams, all stereo, all the same codec, same sample rate
(`src/sources/soundsourcestem.cpp`). This means:

- No changes needed to `EngineDeck`'s stem-channel splitting, mute/solo
  controls, `WTrackStemMenu`, `WStemLabel`, or waveform stem rendering.
- The only new "bridging" logic needed is telling Mixxx *which file* to open
  for playback (see §4.4).

Codec choice: **ALAC** (Apple Lossless), muxed into an MP4/M4A container via
FFmpeg's `libavformat`/`libavcodec` (FFmpeg is already a Mixxx dependency
for STEM support). ALAC was chosen over FLAC-in-MP4 because FFmpeg's
MP4 muxer support for FLAC is comparatively new/version-dependent, whereas
ALAC-in-MP4 is a long-standing, universally supported combination in
FFmpeg's muxer, matching the native `.stem.mp4` format's own codec family.

The container also carries a **STEM manifest atom** (`moov/udta/stem`,
the JSON blob Native Instruments' format uses, written by
`addStemManifestAtom()` in `stemcontainerwriter.cpp`). This is not
optional: `SoundSource::getTypeFromFile()` only classifies an MP4 as a STEM
file if `StemInfoImporter::hasStemAtom()` finds that atom, so without it
the generated container is opened by the plain FFmpeg source as an ordinary
stereo file (its first stream, the premix) and no stem channels ever reach
the engine. libavformat has no API for writing arbitrary atoms, so it is
spliced into the finished file afterwards: `moov` is read back, the atom is
added to its `udta` child (created if absent), the box sizes are grown and
`moov` is rewritten. That is only safe because FFmpeg's MP4 muxer writes
`moov` last — the writer verifies this and refuses to patch the file
otherwise, since growing a leading `moov` would invalidate the sample
offsets stored inside it.

The premix stream is the **original decoded track audio**, not a re-mix of
the 4 separated stems — this avoids compounding any AI-separation
imperfections into the "all stems on" playback path and keeps parity with
how NI's own STEM format treats the premix as the authoritative master.

### 3.3 Storage: per-track folder under the Mixxx settings directory

Chosen location: `<MixxxSettingsPath>/stems/<trackId>/stems.stem.m4a`.

Rejected alternative: writing into a folder next to the original track
file. Rejected because:
- Music library folders are frequently read-only, on removable media, or
  on network shares.
- Keeps generated/derived data clearly separated from the user's own
  library content (consistent with how Mixxx already treats other derived,
  non-essential caches).

The path is persisted per-track in the library DB (`library.stems_path`)
because — unlike *embedded* STEM metadata, which is cheaply re-derived from
the track file's own bytes on every load — there is no way to cheaply
"re-discover" that AI separation has already happened without this
persisted pointer, and the separation itself is an expensive job that
shouldn't be silently redone.

### 3.4 Background processing

Runs via `QtConcurrent::run()` on a controller-owned, single-threaded
`QThreadPool` (see §3.6) + `QFutureWatcher`, matching an existing
precedent in the codebase (`OverviewCache`'s background waveform-overview
generation) rather than extending the more heavyweight
`AnalyzerThread`/`WorkerThread` infrastructure used for beat/key detection.
This was a scope/complexity trade-off: `AnalyzerThread` has its own
SPSC-queue-based thread lifecycle protocol designed for the analyzer
pipeline; stem separation doesn't need to participate in that pipeline (it's
not blocking track-load analysis), so a simpler, self-contained background
task was preferred.

### 3.5 Metadata safety

`SoundSourceProxy(TrackPointer)` (used for metadata import/export, "Import
From File Tags", etc.) is **deliberately left unchanged** — it always
resolves to the track's own file. A separate `SoundSourceProxy::forPlayback()`
factory is used **only** at the two call sites that decode audio samples
(`CachingReaderWorker::loadTrack` and `AnalyzerThread`), so redirecting
playback to the generated STEM container can never accidentally cause Mixxx
to read/write tags against the AI-generated file instead of the user's real
track file.

The analyzer has to follow the same redirect as playback, for the same
reason a pre-authored STEM file is analyzed as a STEM file: analyzing the
original stereo file while the engine plays the 8-channel container stores
a stereo waveform (and stereo stream info) for the track, which hides the
stem UI again right after a separation finishes, and produces a beatgrid
for audio that is not what gets played (the ALAC container's first sound is
slightly offset from the original file's).

### 3.6 Hardware acceleration (GPU/NPU execution providers)

The model is run on the fastest ONNX Runtime execution provider the local
build actually ships with, instead of the default CPU one. `StemSeparator`
takes a `StemSeparatorOptions` struct and configures the session
accordingly (`configureExecutionProviders()` in `stemseparator.cpp`):

- **Selection order** (`acceleration = "auto"`, the default): CUDA →
  DirectML → CoreML → CPU. Each candidate is skipped unless ONNX Runtime
  reports it in `Ort::GetAvailableProviders()` *and* it can actually be
  appended to the session options; failures are logged and fall through to
  the next candidate, so a build without any GPU provider behaves exactly
  as before. The preference can be pinned to a specific provider via the
  `[StemSeparation] / Acceleration` config key (`cpu`, `coreml`, `cuda`,
  `directml`).
- **CoreML** is requested in `MLProgram` format (Core ML 5+), which maps
  far more of htdemucs onto the GPU/Neural Engine than the legacy
  `NeuralNetwork` format, with `MLComputeUnits=ALL` by default
  (configurable via `[StemSeparation] / CoreMLComputeUnits`). On an M4 Pro,
  ONNX Runtime reports 1479 of the graph's 1504 nodes as CoreML-supported.
- **CUDA** uses `OrtCudnnConvAlgoSearchExhaustive`: the one-off algorithm
  search cost is amortized over the many segments of a single track. This
  path is written but **untested** — no NVIDIA hardware was available.
- **DirectML** is compiled in only when `dml_provider_factory.h` is present
  (i.e. a Windows ONNX Runtime build), and disables memory-pattern
  optimization and parallel execution as that provider requires. Also
  **untested**.

Two costs come with the accelerated providers, both mitigated:

- **Ahead-of-time model compilation.** Core ML compiles the ONNX graph into
  its own format on first session creation: 33–54 s, and ~635 MB on disk.
  `StemSeparatorOptions::modelCacheDirectory` points Core ML at
  `<settings>/stems/modelcache`, which cuts subsequent session creation to
  ~9 s. Without it, that compilation would be repeated for *every*
  separation job.
- **Per-job model loading.** The remaining ~9 s load is paid once per
  session rather than once per track: `StemSeparatorCache`
  (`stemseparator.h`) hands every job a shared, already-loaded
  `StemSeparator`. This is safe because a loaded `demucs_model` is
  read-only and `Ort::Session::Run()` is thread-safe; only the lazy load
  itself is mutex-guarded. The controller drops its reference to the cache
  5 minutes after the last job was started, so an idle Mixxx doesn't hold
  on to the model's few hundred MB indefinitely.

Separation jobs also run on a dedicated `QThreadPool` with
`maxThreadCount == 1`. Selecting 20 tracks in the library previously fanned
out one job per core, all competing for the same accelerator and each
holding a fully decoded track plus inference tensors (measured peak RSS for
a single 4-minute job: ~4.2 GB on CoreML, ~7.0 GB on CPU, debug build).
Queuing them is both faster overall and far less memory-hungry.

Options that were measured and *not* adopted because they made no
difference on this hardware (≤5%, within run-to-run noise): Core ML's
`AllowLowPrecisionAccumulationOnGPU` and
`SpecializationStrategy=FastPrediction`, and `MLComputeUnits=CPUAndGPU`
(i.e. excluding the Neural Engine) which was slightly *slower* than `ALL`.

## 4. Implementation details

### 4.1 New files

```
lib/demucsonnx/                             vendored htdemucs/ONNX Runtime inference library (MIT)
  README.mixxx.md                           provenance/commit notes
  demucs.hpp, model_apply.cpp, model_inference.cpp,
  dsp.{hpp,cpp}, tensor.hpp

src/track/stemseparation/
  stemseparator.{h,cpp}                     wraps demucs.onnx; execution-provider selection,
                                             model loading (+ shared StemSeparatorCache), inference
  stemcontainerwriter.{h,cpp}                writes the 5-stream ALAC .stem.m4a container
  stemseparationcontroller.{h,cpp}          singleton; background job orchestration,
                                             per-track output folder, DB update trigger
```

### 4.2 Build system (`CMakeLists.txt`)

New option `STEM_SEPARATION` (default `OFF`), which:
- Requires `STEM` to also be enabled (fatal error otherwise), since it's an
  extension of that feature.
- `find_package(Eigen3 REQUIRED)` and `find_package(onnxruntime REQUIRED)`.
  On macOS, the latter is satisfied by Homebrew's `onnxruntime` formula
  (which installs a CMake package config exporting an
  `onnxruntime::onnxruntime` imported target); other platforms would use
  a vcpkg port or system package providing the same CMake config.
- Builds `lib/demucsonnx/*` as a separate static library target
  (`DemucsOnnx`), mirroring the existing `QueenMaryDsp` pattern for vendored
  code.
- Defines `__STEM_SEPARATION__` (mirroring the existing `__STEM__` macro
  convention) and adds the new `src/track/stemseparation/*.cpp` sources to
  `mixxx-lib`.

Unlike demucs.cpp, demucs.onnx doesn't use OpenMP, so the previous
workaround linking `OpenMP::OpenMP_CXX` into `mixxx-test` (to keep its
reused precompiled header's compiler flags in sync with `mixxx-lib`) is no
longer needed and has been removed.

### 4.3 Library database schema

- `res/schema.xml`: new **revision 41** adds `library.stems_path TEXT
  DEFAULT ''`. This column exists unconditionally in the schema (schema
  version must be consistent across all builds sharing a library file,
  regardless of whether a given build was compiled with
  `STEM_SEPARATION`); only the C++ code that reads/writes it is
  `#ifdef __STEM_SEPARATION__`-gated.
- `src/database/mixxxdb.cpp`: `kRequiredSchemaVersion` bumped `40 → 41`.
- `src/track/trackrecord.h/.cpp`: new `stemsPath` property on `TrackRecord`
  (via the existing `MIXXX_DECL_PROPERTY` macro), included in
  `operator==`.
- `src/library/dao/trackdao.cpp`: `stems_path` added to the INSERT/UPDATE
  column lists, `bindTrackLibraryValues()`, and the `getTrackById()` column
  populator table (new `setTrackStemsPath()` callback).

### 4.4 `Track` class (`src/track/track.h/.cpp`)

- `getStemsPath()` / `setStemsPath(QString)` (new, `__STEM_SEPARATION__`
  only). `setStemsPath()`:
  - Updates `TrackRecord::stemsPath`.
  - Populates/clears `m_stemInfo` with fixed drums/bass/other/vocals
    labels+colors (so `hasStem()`/`getStemInfo()` — used throughout the
    existing STEM UI — work unmodified regardless of whether the stem data
    came from an embedded atom or from AI separation).
  - Marks the track dirty and emits the existing `stemsUpdated()` signal.
- `Track::updateStreamInfoFromSource()` skips its stem-info re-import when
  `stemsPath` is set. That re-import runs whenever an audio source is
  opened and re-reads stem info from *the track's own file*, which it
  attempts for every mp4/m4a track (`StemInfoImporter::maybeStemFile()`
  accepts any MP4-ish MIME type). For an AI-separated m4a it finds no
  manifest and clears the stem info that `setStemsPath()` had just
  installed — i.e. the stems disappeared again the moment the deck loaded
  the track.

### 4.5 Playback routing (`src/sources/soundsourceproxy.{h,cpp}`)

- New private constructor `SoundSourceProxy(TrackPointer, const QUrl&
  urlOverride)` and a new public static factory
  `SoundSourceProxy::forPlayback(TrackPointer)` (`__STEM_SEPARATION__`
  only): resolves to `track->getStemsPath()` if set, otherwise identical to
  the existing `SoundSourceProxy(TrackPointer)`.
- `forPlayback()` falls back to the track's own file (with a warning) if
  the configured container no longer exists, since nothing stops a user
  from deleting folders under `<settings>/stems/` — without this, such a
  track fails to load at all.
- `src/engine/cachingreader/cachingreaderworker.cpp`: `loadTrack()` and
  `src/analyzer/analyzerthread.cpp`: the analysis loop use
  `SoundSourceProxy::forPlayback()` instead of the plain constructor when
  `__STEM_SEPARATION__` is defined. These are the **only** two call sites
  changed; all metadata import/export call sites are untouched (see
  §3.5).

### 4.6 UI triggers

- `src/mixer/basetrackplayer.{h,cpp}`: new per-deck `ControlPushButton`
  `[ChannelN], stem_separate` (skinnable like any other deck button) →
  `StemSeparationController::separateTrack(loadedTrack)`. Also listens for
  `StemSeparationController::separationFinished` and reloads the deck if it
  matches the currently loaded track, so the newly generated STEM container
  gets picked up immediately.
- `src/widget/wtrackmenu.{h,cpp}`: "Separate Stems (AI)" entry added to the
  existing `Analyze` submenu, operating on all currently selected tracks.

### 4.8 Progress reporting

`StemSeparationController::separationProgress(trackId, progress,
statusMessage)` is driven by `QPromise::setProgressValueAndText()` inside
the `QtConcurrent` job, re-emitted on the owning thread by the existing
`QFutureWatcher` — the same mechanism Qt provides for this, so no manual
cross-thread marshalling is needed. The 0..1 scale covers the whole job,
not just inference: model load (0.00), decode (0.05), `demucs.onnx`
inference scaled into 0.10–0.90, container write (0.90). demucs.onnx's own
status strings name internal inference stages, so only its numeric
progress is used; the user-visible stage text comes from Mixxx and is
translatable.

Two consumers:

- `WOverview` (`src/widget/woverview.{h,cpp}`) draws the progress for the
  track shown in that deck's waveform overview: a progress line plus
  `"<stage>... <n>%"` text, mirroring the existing analyzer-progress
  overlay (`drawAnalyzerProgress()`) that every skin already gets for
  free. This was chosen over a new widget next to the stem mute/solo
  controls because it needs no changes to any skin, and the overview is
  already Mixxx's place for "long background job running on this track".
- `src/mixer/basetrackplayer.{h,cpp}`: a read-only per-deck
  `ControlObject` `[ChannelN], stem_separate_progress` (0..1 while
  running, `-1` while idle), the natural companion to the existing
  `[ChannelN], stem_separate` push button, so skins and controller
  mappings can drive their own indicator.

### 4.7 `StemSeparationController` lifecycle

- A `Singleton<StemSeparationController>` (same pattern as `OverviewCache`),
  created/destroyed in `src/coreservices.cpp` alongside other app-lifetime
  singletons.
- Reads the model file path from config key `[StemSeparation] /
  ModelFilePath` (see §7 — no preferences UI for this yet), defaulting to
  `~/.mixxx-demucs/weights/htdemucs.onnx` if unset
  (`defaultStemModelFilePath()` in `stemseparationcontroller.cpp`), plus
  the acceleration settings described in §3.6 (`Acceleration`,
  `CoreMLComputeUnits`), also config-key-only for now.
- Creates `<settings>/stems/modelcache` for the compiled-model cache
  (§3.6) at construction; if that fails it logs a warning and runs without
  a cache rather than failing.
- Owns the `StemSeparatorCache` holding the shared loaded model and the
  `QTimer` that releases it after 5 idle minutes.
- `separateTrack()`: decodes the track to 44.1 kHz stereo via
  `SoundSourceProxy(pTrack).openAudioSource()`, runs `StemSeparator::separate()`
  on the single-threaded separation pool, writes the 5-stream container via
  `writeStemContainer()`, and — back on the owning thread, via
  `QFutureWatcher::finished` — calls `Track::setStemsPath()` on success.

## 5. Stem naming/ordering convention

Fixed order used throughout (`StemSeparationTarget` enum in
`stemseparator.h`), matching demucs.onnx's own 4-source model output order:

| Index | Target | File/label   |
|-------|--------|--------------|
| 0     | Drums  | `drums`      |
| 1     | Bass   | `bass`       |
| 2     | Other  | `other`      |
| 3     | Vocals | `vocals`     |

## 6. Testing performed

No human has tested this feature with real audio, model weights, or DJ
hardware. The following automated/manual verification was done during
development (originally against demucs.cpp; re-verified against
demucs.onnx after the migration described in §3.1):

1. **Vendored demucs.onnx library**: compiled all 3 `.cpp` files standalone
   against Homebrew Eigen3 + Homebrew `onnxruntime` 1.32.0 on macOS, linked
   into a static library.
2. **End-to-end inference smoke test (new, real weights)**: converted the
   *official* Meta `htdemucs` PyTorch checkpoint to ONNX using upstream
   demucs.onnx's own `scripts/convert-pth-to-onnx.py` (see §7 for the full
   procedure) and ran a standalone test program, linked against the
   vendored library and Homebrew's `onnxruntime`, that loads the resulting
   `htdemucs.onnx` and calls `demucsonnx::demucs_inference()` directly:
   - Loads successfully; `model.nb_sources` reports `4`, confirming the
     ONNX graph's input/output tensor shapes line up with what
     `demucs_segment_buffers` expects (two inputs — time-domain `xt` and
     spectrogram `x` — and two outputs, by positional index, not name).
   - Ran real ONNX Runtime inference (not a stub/mock) on both 1 second of
     silence and 30 seconds of random noise audio; produced correctly
     shaped 4-source stereo output in both cases.
   - **Timing**: a 30-second clip separated in ~7.2s on an Apple M4 Pro CPU
     core (single `CPUExecutionProvider`, no GPU/NPU provider configured) —
     i.e. faster than realtime and a large improvement over demucs.cpp,
     though still not instantaneous; this is the whole motivation for the
     migration in §3.1.
3. **Execution-provider benchmark (new)**: a standalone program linked
   against Homebrew `onnxruntime` 1.32.0 that creates sessions with
   different providers/options and times one htdemucs segment
   (`[1,2,343980]` + `[1,4,2048,336]` inputs) per session, on an Apple M4
   Pro (14 cores, 48 GB):

   | Provider / options | session init (cold / warm cache) | per segment (steady state) |
   |---|---|---|
   | CPU (ORT defaults) | 0.18 s | 1.18–1.21 s |
   | CoreML MLProgram, `MLComputeUnits=ALL` | 54.4 s / 9.1 s | 0.26–0.27 s |
   | CoreML MLProgram, `CPUAndGPU` | 32.8 s / 0.17 s | 0.33 s |
   | CoreML MLProgram, `ALL` + low-precision GPU accumulation | 53.5 s | 0.25–0.27 s |
   | CoreML MLProgram, `ALL` + `FastPrediction` | 53.8 s | 0.25–0.28 s |

   The first run after session creation is consistently slower (~0.7–8 s)
   than the steady state. These numbers drove the choices in §3.6.
4. **Accelerated path through Mixxx's own code (new)**: a harness linked
   against the real `mixxx-lib` (same compile/link flags as the `mixxx`
   executable, with `main.cpp` swapped out) driving
   `mixxx::StemSeparatorCache` / `mixxx::StemSeparator::separate()` with
   the real weights on synthetic multi-tone audio:
   - Reports provider `CoreML (ALL)` under `acceleration = "auto"`, i.e.
     the selection logic picks the accelerated provider on this machine,
     and `CPU` when pinned to `"cpu"`.
   - 240 s of audio: **15.1 s on CoreML vs 49.8 s on CPU** (16x vs 5x
     realtime); 20 s of audio: 1.83–1.91 s vs 4.89 s. Peak RSS 4.2 GB
     (CoreML) vs 7.0 GB (CPU), debug build.
   - `StemSeparatorCache` returns the identical instance on the second
     call (0 ms vs 9.1 s), confirming the per-job model load is paid once.
   - All 4 stems come back at the expected length with plausible,

     differing RMS levels (the synthetic bass/mid tones land mostly in
     `bass`/`other`, near-silence in `drums`/`vocals`). This is a
     sanity check, **not** a separation-quality measurement.
5. **Generated container is recognized and played as a STEM file (new)**:
   a harness linked against the real `mixxx-lib` writes a container with
   `mixxx::writeStemContainer()` (5 synthetic tone streams), then checks
   the full chain:
   - `ffprobe`: 5 stereo ALAC streams at 44100 Hz.
   - `StemInfoImporter::hasStemAtom()` → true, and `importStemInfos()`
     parses the 4 expected labels/colors back out of the manifest.
   - `Track::setStemsPath()` → `hasStem()` true with 4 stem infos.
   - `SoundSourceProxy::forPlayback(track).openAudioSource()` returns an
     **8-channel** source (4 stereo stems), and `hasStem()` is *still*
     true afterwards.

   Before the two fixes described in §3.2/§4.4 this same harness reported a
   2-channel source and `hasStem() == false` after opening — which is
   exactly the "stems don't load after separating" behavior observed in
   the running app.
6. **STEM container muxer**: wrote a standalone (Qt-free) replica of
   `stemcontainerwriter.cpp`'s FFmpeg muxing logic, compiled/linked against
   Homebrew FFmpeg, generated a real 5-stream `.stem.m4a` file from
   synthetic sine-wave audio, and verified with `ffprobe` that it has
   exactly 5 stereo ALAC streams at 44100 Hz (matching what
   `SoundSourceSTEM::tryOpen()` requires) and that the streams are audibly
   distinct (decoded and byte-compared two streams). Unaffected by the
   demucs.cpp→demucs.onnx migration; not re-run this session.
7. **Full real build**, reconfigured and rebuilt after the migration with
   `-DSTEM=ON -DSTEM_SEPARATION=ON` and `CMAKE_PREFIX_PATH` pointed at
   Homebrew's `onnxruntime`:
   - `mixxx-lib` (includes all new code) — succeeds.
   - `mixxx` (the full executable) — links successfully, runs, and reports
     `Mixxx 2.7.0-alpha` via `--version`.
   - `mixxx-test` — builds and links successfully (the demucs.cpp-era
     OpenMP/PCH workaround in §4.2 was removed, not just left in place,
     since demucs.onnx has no OpenMP dependency to go out of sync on).
8. **Automated tests**: ran `SchemaManagerTest.*` (confirms schema
   migration through revision 41 still applies cleanly), `TrackDAOTest.*`,
   and every suite matching `*Stem*` — all 30 tests pass. The full test
   binary was *not*
   re-run to completion this session (it's known from the original
   demucs.cpp-era pass to take 10+ minutes due to an unrelated controller
   mapping-validation suite, and to hit one pre-existing, unrelated crash
   in `QmlLibraryProxyTest` — see the original testing notes in git history
   for that pass's full detail).

No test exercises actual separation output quality (vs. ground truth
stems), audible playback of a generated STEM file through the engine,
or the UI buttons in a running application — these require a human to
click through the app with real model weights, which per `AGENTS.md` must
happen before any PR is submitted.

## 7. Known gaps / follow-ups for a human contributor

- **No model weights bundled/auto-downloaded.**
  `StemSeparationController::isAvailable()` just checks that a configured
  file path exists; there is no preferences UI to set it and no in-app
  download. The config key now defaults to
  `~/.mixxx-demucs/weights/htdemucs.onnx`, which a developer can populate
  by converting the **official** Meta `htdemucs` PyTorch checkpoint to
  ONNX using upstream demucs.onnx's own conversion script (already done in
  this working tree, for local testing, using the exact procedure below —
  no third-party pre-converted weights were used; a candidate third-party
  source, the `StemSplitio` org on Hugging Face, was evaluated and
  rejected: zero-download, newly created, promotional-content-heavy repos
  with no independent verification, and a tensor input/output contract
  that doesn't match this vendored library's two-input/two-output ONNX
  graph expectations anyway):
  1. `git clone --depth 1 https://github.com/sevagh/demucs.onnx` (cloned
     into an isolated scratch directory, not this repo).
  2. Create an isolated Python (3.12+) virtualenv and install:
     `torch torchaudio einops julius lameenc openunmix pyyaml tqdm
     dora-search diffq onnx onnxscript`, then
     `pip install --no-deps -e demucs.onnx/demucs-for-onnx` (the patched,
     vendored copy of Meta's `demucs` package that upstream ships, with
     STFT/iSTFT moved outside the PyTorch graph so it's exportable).
  3. Run `python demucs.onnx/scripts/convert-pth-to-onnx.py <dest_dir>`.
     This downloads the real pretrained checkpoint directly from Meta's
     own CDN (`https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/
     955717e8-8726e21a.th`, ~80 MB) via `demucs.pretrained.get_model()`,
     then runs `torch.onnx.export()` and writes `htdemucs.onnx`.
  4. The exporter (on current PyTorch/onnxscript versions) saves large
     weight tensors to an external `htdemucs.onnx.data` file alongside the
     small graph-only `.onnx` file. Since `StemSeparator` loads the model
     from an in-memory byte buffer (`demucsonnx::load_model(const
     std::vector<char>&, ...)`, not a file path), external data isn't
     resolvable — inline it into one self-contained file:
     `onnx.save(onnx.load(path, load_external_data=True), out_path)`
     (fits comfortably under the 2 GB protobuf limit at ~170 MB).
  5. Verified structurally with the `onnx` Python package (no `onnxruntime`
     wheel was available for this environment's Python 3.15 build, so this
     checked the graph proto directly rather than running it in Python):
     inputs `input` `[1,2,343980]` and `x` `[1,4,2048,336]`; outputs
     `output` `[1,4,4,2048,336]` and (name varies with exporter version)
     `[1,4,2,343980]` — exactly the `xt`/`x` input pair and `x_out`/`xt_out`
     output pair `lib/demucsonnx/demucs.hpp`'s `demucs_segment_buffers`
     expects, by position. `nb_sources` (from `output[1]`) is `4`, as
     required.
  6. Verified end-to-end in C++ instead (see §6 item 2) since Python-side
     `onnxruntime` wasn't installable here.

  A real implementation still needs: a preferences page entry to override
  the model path, an in-app download flow with size/license disclosure
  (pointing at a *verified* source for pre-converted weights, or at this
  conversion procedure), and documentation of the same conversion path for
  other htdemucs variants (6-source, fine-tuned bag-of-4-models) — see
  `lib/demucsonnx/README.mixxx.md` and upstream's
  `scripts/convert-pth-to-onnx.py --six-source` / `--ft-*` flags. Note the
  fine-tuned bag-of-4-models variant doesn't fit this controller's
  single-model assumption without further changes (it's 4 separate ONNX
  sessions averaged together, not 1).
- **New runtime dependency: ONNX Runtime.** Unlike Eigen3/OpenMP
  (demucs.cpp-era), `onnxruntime` is a much larger prebuilt binary library
  (tens of MB) that isn't yet part of Mixxx's vcpkg dependency bundle
  (`tools/*_buildenv.sh`) or guaranteed to be packaged on every target
  Linux distro. A real implementation needs to either add it to Mixxx's
  vcpkg manifest/buildenv bundles for all platforms, or confirm it's
  acceptably available via system packages (Homebrew has it on macOS;
  most major Linux distros package `onnxruntime` or close variants, but
  this hasn't been verified for all of Mixxx's supported distros).
- **Acceleration settings have no preferences UI.** `[StemSeparation] /
  Acceleration` and `/ CoreMLComputeUnits` (§3.6) are config-file-only,
  like the model path. A real implementation should expose them (at least
  an "use hardware acceleration" toggle, ideally the detected provider
  name) in the preferences page that will also host the model path.
- **CUDA and DirectML paths are untested.** Only the CoreML and CPU
  providers were exercised; the NVIDIA and Windows paths are written from
  the ONNX Runtime API contract and need verification on real hardware,
  including whether `OrtCudnnConvAlgoSearchExhaustive` is the right
  trade-off there and whether ONNX Runtime builds shipped on those
  platforms actually include the providers.
- **Compiled-model cache is never cleaned up.** Core ML's cache under
  `<settings>/stems/modelcache` is ~635 MB for htdemucs and is not
  invalidated if the model file changes — ONNX Runtime's cache key is a
  hash of the model, so a changed model simply adds another entry rather
  than replacing the old one. Needs the same management/cleanup story as
  the generated stem containers below.
- **First separation after installing a model is slow.** The one-off Core
  ML compilation (33–54 s on an M4 Pro) happens inside the first
  separation job, where the progress UI reports it as the "Loading
  separation model..." stage at 0% with no further updates. Worth either
  doing eagerly/in the background at startup, or at least reporting as a
  distinct, explained stage.
- **Progress is only shown for tracks loaded on a deck.** Separation
  started from the library context menu for tracks that are not loaded on
  any deck runs without a visible indicator (see §4.8); it would need a
  library-side display comparable to `DlgAnalysis`'s progress bar.
- **Re-analysis after separation is implicit.** Once separation finishes,
  the deck reloads and the stored (stereo) analysis no longer matches the
  8-channel container, which is what currently causes Mixxx to re-analyze
  and produce a stem waveform. Nothing explicitly requests that
  re-analysis, so a track separated from the library context menu while
  not loaded on a deck keeps its stereo waveform until it is next loaded.
  A real implementation should queue the re-analysis explicitly when
  `separationFinished` fires.
- **Playback interruption.** When separation finishes for a track that's
  *currently playing* on a deck, the deck is unloaded and reloaded to pick
  up the new STEM container, which stops playback. A smoother approach
  (e.g. deferring the reload until the track is stopped/ejected, or hot
  -swapping the audio source) would need more engine-level work.
- **Containers generated before the manifest-atom fix are unusable.**
  Any `.stem.m4a` written by an earlier build lacks the
  `moov/udta/stem` atom (§3.2) and will keep loading as a plain stereo
  file. There is no detection or migration for this — the affected tracks
  have to be separated again, and the stale folders under
  `<settings>/stems/<trackId>/` deleted by hand.
- **No cleanup/management UI.** Generated STEM containers under
  `<settings>/stems/<trackId>/` are never deleted (e.g. when a track is
  removed from the library, or to let a user regenerate/discard them).
- **Untested at runtime.** As noted in §6, nothing here has been exercised
  with real weights/audio or a real DJ setup yet.
