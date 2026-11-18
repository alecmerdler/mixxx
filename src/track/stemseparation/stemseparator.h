#pragma once

#include <QColor>
#include <QMutex>
#include <QString>

#include <functional>
#include <memory>
#include <vector>

#include "util/types.h"

namespace mixxx {

// Names and order match the 4 output targets of the htdemucs model used by
// demucs.onnx (lib/demucsonnx). They double as the output filenames written
// by StemSeparator::separate().
enum class StemSeparationTarget {
    Drums = 0,
    Bass = 1,
    Other = 2,
    Vocals = 3,
};

constexpr int kNumStemSeparationTargets = 4;

QString stemSeparationTargetName(StemSeparationTarget target);

// Display name of a target, as written into the generated container's STEM
// manifest and shown by the stem UI.
QString stemSeparationTargetLabel(StemSeparationTarget target);

// Colorblind-friendly palette, matching the default colors used for
// imported STEM file metadata in steminfoimporter.cpp.
QColor stemSeparationTargetColor(StemSeparationTarget target);

// Hardware acceleration settings for the ONNX Runtime session that runs the
// model. The defaults pick the fastest execution provider the ONNX Runtime
// build actually ships with (GPU/NPU where available, CPU otherwise).
struct StemSeparatorOptions {
    // Execution provider preference:
    //   "auto"     - try the accelerated providers this build supports,
    //                in descending order of expected throughput, and fall
    //                back to CPU if none of them can be created.
    //   "cpu"      - force the CPU execution provider.
    //   "coreml"   - macOS GPU/Neural Engine via Core ML.
    //   "cuda"     - NVIDIA GPUs.
    //   "directml" - any DirectX 12 GPU on Windows.
    // Unknown values are treated as "auto".
    QString acceleration = QStringLiteral("auto");

    // Core ML only: which compute units Core ML may dispatch to, one of
    // "ALL", "CPUAndGPU", "CPUAndNeuralEngine" or "CPUOnly". "ALL" lets
    // Core ML use the Neural Engine, the GPU and the CPU.
    QString coreMLComputeUnits = QStringLiteral("ALL");

    // Where providers that compile the model ahead of time (Core ML) may
    // cache the compiled model. Without it, that compilation (tens of
    // seconds for htdemucs) is repeated on every session creation. The
    // directory must exist; it is not created here.
    QString modelCacheDirectory;

    // CPU execution provider only: number of threads used within a single
    // operator. 0 keeps ONNX Runtime's default (one per physical core).
    int intraOpThreads = 0;
};

// Wraps the vendored demucs.onnx inference library to split a decoded track
// into drums/bass/other/vocals stem files using the htdemucs 4-source model,
// run via ONNX Runtime.
//
// This class only performs the separation itself: it does not decode tracks,
// manage the per-track output folder layout, or hook into track loading.
class StemSeparator {
  public:
    using ProgressCallback = std::function<void(float progress, const QString& statusMessage)>;

    // modelFilePath must point to an ONNX-format htdemucs 4-source weights
    // file (see lib/demucsonnx/README.mixxx.md). Loading happens eagerly in
    // the constructor; check isModelLoaded() before calling separate().
    explicit StemSeparator(const QString& modelFilePath,
            const StemSeparatorOptions& options = StemSeparatorOptions());
    ~StemSeparator();

    bool isModelLoaded() const;

    // Human-readable name of the execution provider the session was
    // actually created with, e.g. "CoreML (ALL)" or "CPU". Empty if the
    // model failed to load.
    QString accelerationDescription() const;

    // audioStereo44100 must contain interleaved stereo samples in [-1, 1] at
    // the sample rate required by demucs.onnx (StemSeparator::sampleRate()).
    // Returns the 4 separated stems, each interleaved stereo and the same
    // length as the input, in StemSeparationTarget order, or an empty list
    // on failure.
    std::vector<std::vector<CSAMPLE>> separate(
            const std::vector<CSAMPLE>& audioStereo44100,
            const ProgressCallback& progressCallback = nullptr) const;

    // The fixed sample rate required by the underlying model.
    static int sampleRate();

  private:
    class Impl;
    std::unique_ptr<Impl> m_pImpl;
};

// Shares one loaded StemSeparator between separation jobs.
//
// Creating a StemSeparator is expensive for the accelerated execution
// providers — Core ML has to load (and, the very first time, compile) a
// GPU/Neural Engine copy of the model — while the loaded session itself is
// read-only and safe to run from several threads at once. Jobs therefore
// share a single instance instead of loading the model per track.
class StemSeparatorCache {
  public:
    StemSeparatorCache(const QString& modelFilePath, const StemSeparatorOptions& options);

    // Loads the model on first call and returns the same instance to every
    // later caller. Thread-safe; concurrent callers block until the model
    // is loaded. Returns nullptr if the model could not be loaded.
    std::shared_ptr<const StemSeparator> separator();

  private:
    QMutex m_mutex;
    const QString m_modelFilePath;
    const StemSeparatorOptions m_options;
    std::shared_ptr<const StemSeparator> m_pSeparator;
};

} // namespace mixxx
