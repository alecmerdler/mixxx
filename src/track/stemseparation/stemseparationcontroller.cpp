#include "track/stemseparation/stemseparationcontroller.h"

#include <QDir>
#include <QFileInfo>
#include <QPromise>
#include <QtConcurrentRun>

#include <vector>

#include "moc_stemseparationcontroller.cpp"
#include "sources/audiosource.h"
#include "sources/soundsourceproxy.h"
#include "track/stemseparation/stemcontainerwriter.h"
#include "track/stemseparation/stemseparator.h"
#include "track/track.h"
#include "util/logger.h"

namespace mixxx {

namespace {

const Logger kLogger("StemSeparationController");

const ConfigKey kStemModelFilePathConfigKey(
        QStringLiteral("[StemSeparation]"), QStringLiteral("ModelFilePath"));

// Which ONNX Runtime execution provider to run the model on; see
// StemSeparatorOptions::acceleration for the accepted values.
const ConfigKey kStemAccelerationConfigKey(
        QStringLiteral("[StemSeparation]"), QStringLiteral("Acceleration"));

// Core ML compute units, see StemSeparatorOptions::coreMLComputeUnits.
const ConfigKey kStemCoreMLComputeUnitsConfigKey(
        QStringLiteral("[StemSeparation]"), QStringLiteral("CoreMLComputeUnits"));

// Default location for the ONNX-format htdemucs 4-source weights file,
// matching the layout documented in lib/demucsonnx/README.mixxx.md.
QString defaultStemModelFilePath() {
    return QDir::homePath() +
            QStringLiteral("/.mixxx-demucs/weights/htdemucs.onnx");
}

// Subdirectory of the stems output folder holding the GPU-compiled copies
// of the model produced by execution providers that compile ahead of time
// (Core ML). Compiling htdemucs takes tens of seconds, so caching it keeps
// that cost to a one-off instead of paying it per separation job.
const QString kModelCacheDirectoryName = QStringLiteral("stems/modelcache");

// How long the loaded model is kept in memory after the last separation
// job was started. Loading it again costs seconds on the accelerated
// execution providers, but it also holds on to a few hundred MB.
constexpr int kSeparatorCacheReleaseTimeoutMs = 5 * 60 * 1000;

// QFuture reports progress as an integer range; this is the number of
// steps the overall 0..1 job progress is quantized into.
constexpr int kProgressResolution = 1000;

// Boundaries of the individual phases of a separation job on the overall
// 0..1 progress scale. Model loading and decoding are a small, roughly
// fixed cost compared to inference, which dominates the job.
constexpr float kDecodeStartProgress = 0.05f;
constexpr float kInferenceStartProgress = 0.1f;
constexpr float kWriteStartProgress = 0.9f;

template<typename T>
void reportProgress(QPromise<T>& promise,
        float progress,
        const QString& statusMessage) {
    promise.setProgressValueAndText(
            static_cast<int>(progress * kProgressResolution), statusMessage);
}

// Decodes an entire track to interleaved stereo float samples at the
// sample rate required by the separation model. Returns an empty buffer
// on failure.
std::vector<CSAMPLE> decodeTrackStereo(const TrackPointer& pTrack, int sampleRate) {
    AudioSource::OpenParams config;
    config.setChannelCount(mixxx::audio::ChannelCount::stereo());
    config.setSampleRate(mixxx::audio::SampleRate(sampleRate));

    auto pAudioSource = SoundSourceProxy(pTrack).openAudioSource(config);
    if (!pAudioSource) {
        kLogger.warning() << "Failed to open audio source for" << pTrack->getLocation();
        return {};
    }

    const SINT numFrames = pAudioSource->frameLength();
    if (numFrames <= 0) {
        return {};
    }

    std::vector<CSAMPLE> buffer(static_cast<size_t>(numFrames) * 2);
    auto writable = WritableSampleFrames(
            IndexRange::forward(pAudioSource->frameIndexMin(), numFrames),
            SampleBuffer::WritableSlice(buffer.data(), buffer.size()));
    const auto readable = pAudioSource->readSampleFrames(writable);
    if (readable.frameLength() != numFrames) {
        kLogger.warning() << "Short read while decoding" << pTrack->getLocation();
    }
    return buffer;
}

} // namespace

StemSeparationController::StemSeparationController(UserSettingsPointer pConfig)
        : m_pConfig(std::move(pConfig)) {
    m_modelFilePath = m_pConfig->getValue(
            kStemModelFilePathConfigKey, defaultStemModelFilePath());

    m_separatorOptions.acceleration = m_pConfig->getValue(
            kStemAccelerationConfigKey, m_separatorOptions.acceleration);
    m_separatorOptions.coreMLComputeUnits = m_pConfig->getValue(
            kStemCoreMLComputeUnitsConfigKey, m_separatorOptions.coreMLComputeUnits);

    const QString modelCacheDirectory =
            QDir(m_pConfig->getSettingsPath()).filePath(kModelCacheDirectoryName);
    if (QDir().mkpath(modelCacheDirectory)) {
        m_separatorOptions.modelCacheDirectory = modelCacheDirectory;
    } else {
        kLogger.warning() << "Failed to create" << modelCacheDirectory
                          << "- the model will be recompiled for every separation job";
    }

    m_threadPool.setMaxThreadCount(1);

    m_separatorCacheReleaseTimer.setSingleShot(true);
    m_separatorCacheReleaseTimer.setInterval(kSeparatorCacheReleaseTimeoutMs);
    connect(&m_separatorCacheReleaseTimer, &QTimer::timeout, this, [this]() {
        if (m_pendingTrackIds.isEmpty()) {
            // Any job still holding a reference keeps the model alive
            // until it finishes.
            m_pSeparatorCache.reset();
        } else {
            m_separatorCacheReleaseTimer.start();
        }
    });
}

std::shared_ptr<StemSeparatorCache> StemSeparationController::separatorCache() {
    if (!m_pSeparatorCache) {
        m_pSeparatorCache = std::make_shared<StemSeparatorCache>(
                m_modelFilePath, m_separatorOptions);
    }
    m_separatorCacheReleaseTimer.start();
    return m_pSeparatorCache;
}

StemSeparationController::~StemSeparationController() = default;

bool StemSeparationController::isAvailable() const {
    return !m_modelFilePath.isEmpty() && QFileInfo::exists(m_modelFilePath);
}

bool StemSeparationController::isSeparating(TrackId trackId) const {
    return m_pendingTrackIds.contains(trackId);
}

QString StemSeparationController::stemsOutputDirectory(TrackId trackId) const {
    return QDir(m_pConfig->getSettingsPath())
            .filePath(QStringLiteral("stems/") + trackId.toVariant().toString());
}

void StemSeparationController::separateTrack(TrackPointer pTrack) {
    if (!pTrack || !isAvailable()) {
        return;
    }
    const TrackId trackId = pTrack->getId();
    if (!trackId.isValid() || isSeparating(trackId)) {
        return;
    }
    m_pendingTrackIds.insert(trackId);
    emit separationStarted(trackId);

    const std::shared_ptr<StemSeparatorCache> pSeparatorCache = separatorCache();
    const QString outputDirectory = stemsOutputDirectory(trackId);

    auto pWatcher = std::make_unique<QFutureWatcher<Result>>();
    QFutureWatcher<Result>* pWatcherPtr = pWatcher.get();
    connect(pWatcherPtr, &QFutureWatcher<Result>::finished, this, [this, pWatcherPtr]() {
        handleFutureFinished(pWatcherPtr);
    });
    // QFutureWatcher re-emits the worker thread's progress reports on this
    // (the owning) thread, so no extra marshalling is needed here.
    connect(pWatcherPtr,
            &QFutureWatcher<Result>::progressValueChanged,
            this,
            [this, pWatcherPtr, trackId](int progressValue) {
                emit separationProgress(trackId,
                        static_cast<float>(progressValue) / kProgressResolution,
                        pWatcherPtr->progressText());
            });
    m_pWatchers.push_back(std::move(pWatcher));

    QFuture<Result> future = QtConcurrent::run(&m_threadPool,
            [pTrack, trackId, pSeparatorCache, outputDirectory](
                    QPromise<Result>& promise) {
                promise.setProgressRange(0, kProgressResolution);

                Result result;
                result.pTrack = pTrack;
                result.trackId = trackId;

                reportProgress(promise, 0.0f, tr("Loading separation model..."));
                const std::shared_ptr<const StemSeparator> pSeparator =
                        pSeparatorCache->separator();
                if (!pSeparator) {
                    promise.addResult(result);
                    return;
                }

                reportProgress(promise, kDecodeStartProgress, tr("Decoding track..."));
                const std::vector<CSAMPLE> premix =
                        decodeTrackStereo(pTrack, StemSeparator::sampleRate());
                if (premix.empty()) {
                    promise.addResult(result);
                    return;
                }

                // The status messages emitted by demucs.onnx name internal
                // inference stages, so only its numeric progress is used and
                // mapped onto the inference phase of the overall job.
                const QString separatingMessage = tr("Separating stems...");
                reportProgress(promise, kInferenceStartProgress, separatingMessage);
                const std::vector<std::vector<CSAMPLE>> stems = pSeparator->separate(
                        premix,
                        [&promise, &separatingMessage](
                                float progress, const QString& /*message*/) {
                            reportProgress(promise,
                                    kInferenceStartProgress +
                                            progress *
                                                    (kWriteStartProgress -
                                                            kInferenceStartProgress),
                                    separatingMessage);
                        });
                if (stems.size() != kNumStemSeparationTargets) {
                    promise.addResult(result);
                    return;
                }

                if (!QDir().mkpath(outputDirectory)) {
                    kLogger.warning() << "Failed to create" << outputDirectory;
                    promise.addResult(result);
                    return;
                }
                const QString containerPath = QDir(outputDirectory).filePath(
                        QStringLiteral("stems.stem.m4a"));

                std::vector<std::vector<CSAMPLE>> streams;
                streams.reserve(1 + kNumStemSeparationTargets);
                streams.push_back(premix);
                for (const auto& stem : stems) {
                    streams.push_back(stem);
                }

                reportProgress(promise, kWriteStartProgress, tr("Writing stem file..."));
                if (!writeStemContainer(
                            containerPath, StemSeparator::sampleRate(), streams)) {
                    promise.addResult(result);
                    return;
                }

                reportProgress(promise, 1.0f, QString());
                result.stemsPath = containerPath;
                result.success = true;
                promise.addResult(result);
            });
    pWatcherPtr->setFuture(future);
}

void StemSeparationController::handleFutureFinished(QFutureWatcher<Result>* pWatcher) {
    const Result result = pWatcher->result();
    m_pendingTrackIds.remove(result.trackId);

    if (result.success && result.pTrack) {
        result.pTrack->setStemsPath(result.stemsPath);
    }
    emit separationFinished(result.trackId, result.success);

    std::erase_if(m_pWatchers, [pWatcher](const std::unique_ptr<QFutureWatcher<Result>>& p) {
        return p.get() == pWatcher;
    });
}

} // namespace mixxx
