#pragma once

#include <QFutureWatcher>
#include <QObject>
#include <QSet>
#include <QString>
#include <QThreadPool>
#include <QTimer>

#include <memory>

#include "preferences/usersettings.h"
#include "track/stemseparation/stemseparator.h"
#include "track/track_decl.h"
#include "track/trackid.h"
#include "util/singleton.h"

namespace mixxx {

// Runs AI-based stem separation for tracks in the background (on a
// QtConcurrent-managed worker thread) and stores the result as a
// STEM-compatible container file in a dedicated per-track folder under
// the Mixxx settings directory. On success, the originating Track is
// updated (Track::setStemsPath()) so it gets picked up on next load.
class StemSeparationController : public QObject, public Singleton<StemSeparationController> {
    Q_OBJECT

  public:
    explicit StemSeparationController(UserSettingsPointer pConfig);
    ~StemSeparationController() override;

    // True if a model file is configured and exists on disk.
    bool isAvailable() const;

    bool isSeparating(TrackId trackId) const;

    // No-op if stems are already being separated for this track, or if
    // isAvailable() is false.
    void separateTrack(TrackPointer pTrack);

  signals:
    void separationStarted(TrackId trackId);
    void separationProgress(TrackId trackId, float progress, QString statusMessage);
    void separationFinished(TrackId trackId, bool success);

  private:
    struct Result {
        TrackPointer pTrack;
        TrackId trackId;
        QString stemsPath;
        bool success = false;
    };

    void handleFutureFinished(QFutureWatcher<Result>* pWatcher);

    QString stemsOutputDirectory(TrackId trackId) const;

    // The loaded model is kept around between jobs (see StemSeparatorCache)
    // and released again once no separation has been started for a while,
    // so idle Mixxx sessions don't hold on to it.
    std::shared_ptr<StemSeparatorCache> separatorCache();

    UserSettingsPointer m_pConfig;
    QString m_modelFilePath;
    StemSeparatorOptions m_separatorOptions;
    // Separation jobs are queued rather than run in parallel: they all
    // compete for the same accelerator (and a few GB of RAM each), so
    // running several at once makes every one of them slower.
    QThreadPool m_threadPool;
    std::shared_ptr<StemSeparatorCache> m_pSeparatorCache;
    QTimer m_separatorCacheReleaseTimer;
    QSet<TrackId> m_pendingTrackIds;
    std::vector<std::unique_ptr<QFutureWatcher<Result>>> m_pWatchers;
};

} // namespace mixxx
