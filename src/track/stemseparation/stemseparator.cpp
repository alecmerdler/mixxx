#include "track/stemseparation/stemseparator.h"

#include <QFile>
#include <QMutexLocker>
#include <QtDebug>

#include <algorithm>
#include <string>
#include <unordered_map>

#if defined(__has_include)
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#endif
#endif

#include "demucs.hpp"
#include "dsp.hpp"
#include "util/logger.h"

namespace mixxx {

namespace {

const mixxx::Logger kLogger("StemSeparator");

constexpr int kNumChannels = 2;

// Names as reported by Ort::GetAvailableProviders() for the providers this
// code knows how to configure.
[[maybe_unused]] const char* const kCoreMLProviderName = "CoreMLExecutionProvider";
[[maybe_unused]] const char* const kCudaProviderName = "CUDAExecutionProvider";
[[maybe_unused]] const char* const kDirectMLProviderName = "DmlExecutionProvider";

bool isProviderAvailable(const char* providerName) {
    const std::vector<std::string> available = Ort::GetAvailableProviders();
    return std::find(available.cbegin(), available.cend(), providerName) !=
            available.cend();
}

// Returns a description of the provider on success, or an empty string if it
// could not be enabled (in which case sessionOptions is left usable for the
// next candidate).
QString tryEnableCoreML(Ort::SessionOptions& sessionOptions,
        const StemSeparatorOptions& options) {
    if (!isProviderAvailable(kCoreMLProviderName)) {
        return QString();
    }
    std::unordered_map<std::string, std::string> providerOptions{
            {"MLComputeUnits", options.coreMLComputeUnits.toStdString()},
            // MLProgram (Core ML 5+) supports far more of htdemucs' ops on
            // the GPU/ANE than the legacy NeuralNetwork format.
            {"ModelFormat", "MLProgram"},
    };
    if (!options.modelCacheDirectory.isEmpty()) {
        providerOptions.emplace("ModelCacheDirectory",
                options.modelCacheDirectory.toStdString());
    }
    try {
        sessionOptions.AppendExecutionProvider("CoreML", providerOptions);
    } catch (const Ort::Exception& e) {
        kLogger.warning() << "Could not enable the CoreML execution provider:" << e.what();
        return QString();
    }
    return QStringLiteral("CoreML (") + options.coreMLComputeUnits + QChar(')');
}

QString tryEnableCuda(Ort::SessionOptions& sessionOptions) {
    if (!isProviderAvailable(kCudaProviderName)) {
        return QString();
    }
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    // Spend a little time at the first inference picking the fastest
    // convolution algorithms; every following segment of the same track
    // reuses that choice.
    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cudaOptions.do_copy_in_default_stream = 1;
    try {
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    } catch (const Ort::Exception& e) {
        kLogger.warning() << "Could not enable the CUDA execution provider:" << e.what();
        return QString();
    }
    return QStringLiteral("CUDA");
}

QString tryEnableDirectML(Ort::SessionOptions& sessionOptions) {
#if defined(__has_include)
#if __has_include(<dml_provider_factory.h>)
    if (!isProviderAvailable(kDirectMLProviderName)) {
        return QString();
    }
    // DirectML does not support ONNX Runtime's parallel execution mode or
    // memory pattern optimization.
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
    const OrtStatus* pStatus = OrtSessionOptionsAppendExecutionProvider_DML(
            static_cast<OrtSessionOptions*>(sessionOptions), 0);
    if (pStatus != nullptr) {
        Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(pStatus));
        kLogger.warning() << "Could not enable the DirectML execution provider";
        return QString();
    }
    return QStringLiteral("DirectML");
#endif
#endif
    Q_UNUSED(sessionOptions);
    return QString();
}

// Appends the best execution provider available for the configured
// preference. The CPU provider is always registered implicitly by ONNX
// Runtime as the fallback for operators an accelerated provider can't take,
// so nothing is appended for it here.
QString configureExecutionProviders(Ort::SessionOptions& sessionOptions,
        const StemSeparatorOptions& options) {
    const QString preference = options.acceleration.trimmed().toLower();

    if (preference == QLatin1String("cpu")) {
        return QStringLiteral("CPU");
    }

    QString description;
    if (preference == QLatin1String("coreml")) {
        description = tryEnableCoreML(sessionOptions, options);
    } else if (preference == QLatin1String("cuda")) {
        description = tryEnableCuda(sessionOptions);
    } else if (preference == QLatin1String("directml")) {
        description = tryEnableDirectML(sessionOptions);
    } else {
        if (preference != QLatin1String("auto")) {
            kLogger.warning() << "Unknown acceleration preference" << options.acceleration
                              << "- falling back to automatic selection";
        }
        // Discrete/integrated GPU providers first, then Apple's Core ML,
        // which may itself dispatch to the Neural Engine or the GPU.
        description = tryEnableCuda(sessionOptions);
        if (description.isEmpty()) {
            description = tryEnableDirectML(sessionOptions);
        }
        if (description.isEmpty()) {
            description = tryEnableCoreML(sessionOptions, options);
        }
    }

    if (description.isEmpty()) {
        return QStringLiteral("CPU");
    }
    return description;
}

const QString kTargetNames[kNumStemSeparationTargets] = {
        QStringLiteral("drums"),
        QStringLiteral("bass"),
        QStringLiteral("other"),
        QStringLiteral("vocals"),
};

const QString kTargetLabels[kNumStemSeparationTargets] = {
        QStringLiteral("Drums"),
        QStringLiteral("Bass"),
        QStringLiteral("Other"),
        QStringLiteral("Vocals"),
};

const QColor kTargetColors[kNumStemSeparationTargets] = {
        QColor(0x00, 0x9E, 0x73),
        QColor(0xD5, 0x5E, 0x00),
        QColor(0xCC, 0x79, 0xA7),
        QColor(0x56, 0xB4, 0xE9),
};

} // namespace

QString stemSeparationTargetName(StemSeparationTarget target) {
    return kTargetNames[static_cast<int>(target)];
}

QString stemSeparationTargetLabel(StemSeparationTarget target) {
    return kTargetLabels[static_cast<int>(target)];
}

QColor stemSeparationTargetColor(StemSeparationTarget target) {
    return kTargetColors[static_cast<int>(target)];
}

class StemSeparator::Impl {
  public:
    demucsonnx::demucs_model model{};
    bool modelLoaded = false;
    QString accelerationDescription;
};

StemSeparator::StemSeparator(const QString& modelFilePath, const StemSeparatorOptions& options)
        : m_pImpl(std::make_unique<Impl>()) {
    QFile modelFile(modelFilePath);
    if (!modelFile.open(QIODevice::ReadOnly)) {
        kLogger.warning() << "Failed to open model file" << modelFilePath;
        return;
    }
    const QByteArray modelBytes = modelFile.readAll();
    const std::vector<char> modelData(modelBytes.constBegin(), modelBytes.constEnd());

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.intraOpThreads > 0) {
        sessionOptions.SetIntraOpNumThreads(options.intraOpThreads);
    }
    m_pImpl->accelerationDescription = configureExecutionProviders(sessionOptions, options);

    try {
        m_pImpl->modelLoaded = demucsonnx::load_model(modelData, m_pImpl->model, sessionOptions);
    } catch (const Ort::Exception& e) {
        kLogger.warning() << "Failed to load ONNX Runtime session from" << modelFilePath << ":"
                           << e.what();
        m_pImpl->modelLoaded = false;
        return;
    }
    if (!m_pImpl->modelLoaded) {
        kLogger.warning() << "Failed to load model from" << modelFilePath;
    } else if (m_pImpl->model.nb_sources == kNumStemSeparationTargets) {
        kLogger.info() << "Loaded" << modelFilePath << "- running on"
                       << m_pImpl->accelerationDescription;
    } else if (m_pImpl->model.nb_sources != kNumStemSeparationTargets) {
        kLogger.warning() << "Model at" << modelFilePath
                           << "is not a 4-source model, which is required "
                              "for Mixxx's stem slots";
        m_pImpl->modelLoaded = false;
    }
}

StemSeparator::~StemSeparator() = default;

bool StemSeparator::isModelLoaded() const {
    return m_pImpl->modelLoaded;
}

QString StemSeparator::accelerationDescription() const {
    return m_pImpl->modelLoaded ? m_pImpl->accelerationDescription : QString();
}

int StemSeparator::sampleRate() {
    return demucsonnx::SUPPORTED_SAMPLE_RATE;
}

std::vector<std::vector<CSAMPLE>> StemSeparator::separate(
        const std::vector<CSAMPLE>& audioStereo44100,
        const ProgressCallback& progressCallback) const {
    if (!isModelLoaded()) {
        return {};
    }
    if (audioStereo44100.empty() || audioStereo44100.size() % kNumChannels != 0) {
        kLogger.warning() << "Input buffer is empty or not interleaved stereo";
        return {};
    }

    const qint64 numFrames = static_cast<qint64>(audioStereo44100.size() / kNumChannels);

    Eigen::MatrixXf audio(kNumChannels, numFrames);
    for (qint64 frame = 0; frame < numFrames; ++frame) {
        for (int channel = 0; channel < kNumChannels; ++channel) {
            audio(channel, frame) =
                    audioStereo44100[static_cast<size_t>(frame) * kNumChannels + channel];
        }
    }

    demucsonnx::ProgressCallback nativeCallback =
            [&progressCallback](float progress, const std::string& message) {
                if (progressCallback) {
                    progressCallback(progress, QString::fromStdString(message));
                }
            };

    const Eigen::Tensor3dXf targets =
            demucsonnx::demucs_inference(m_pImpl->model, audio, nativeCallback);

    std::vector<std::vector<CSAMPLE>> stems(
            kNumStemSeparationTargets, std::vector<CSAMPLE>(audioStereo44100.size()));
    for (int target = 0; target < kNumStemSeparationTargets; ++target) {
        for (qint64 frame = 0; frame < numFrames; ++frame) {
            for (int channel = 0; channel < kNumChannels; ++channel) {
                stems[target][static_cast<size_t>(frame) * kNumChannels + channel] =
                        targets(target, channel, frame);
            }
        }
    }
    return stems;
}

StemSeparatorCache::StemSeparatorCache(
        const QString& modelFilePath, const StemSeparatorOptions& options)
        : m_modelFilePath(modelFilePath), m_options(options) {
}

std::shared_ptr<const StemSeparator> StemSeparatorCache::separator() {
    const QMutexLocker locked(&m_mutex);
    if (!m_pSeparator) {
        auto pSeparator = std::make_shared<const StemSeparator>(m_modelFilePath, m_options);
        if (!pSeparator->isModelLoaded()) {
            return nullptr;
        }
        m_pSeparator = std::move(pSeparator);
    }
    return m_pSeparator;
}

} // namespace mixxx
