#include "track/stemseparation/stemcontainerwriter.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100) // FFmpeg 5.1
#include <libavutil/channel_layout.h>
#endif
}

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtDebug>
#include <QtEndian>
#include <QtGlobal>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "track/stemseparation/stemseparator.h"
#include "util/logger.h"

namespace mixxx {

namespace {

const Logger kLogger("StemContainerWriter");

constexpr int kNumChannels = 2;

void setStereoChannelLayout(AVCodecContext* pCodecContext) {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100) // FFmpeg 5.1
    av_channel_layout_default(&pCodecContext->ch_layout, kNumChannels);
#else
    pCodecContext->channels = kNumChannels;
    pCodecContext->channel_layout = AV_CH_LAYOUT_STEREO;
#endif
}

struct EncoderStream {
    AVCodecContext* pCodecContext = nullptr;
    AVStream* pStream = nullptr;
    AVFrame* pFrame = nullptr;
    int64_t nextPts = 0;
};

bool sendAndWritePacket(AVFormatContext* pFormatContext,
        AVCodecContext* pCodecContext,
        AVStream* pStream,
        AVFrame* pFrame,
        AVPacket* pPacket) {
    int ret = avcodec_send_frame(pCodecContext, pFrame);
    if (ret < 0) {
        kLogger.warning() << "avcodec_send_frame failed:" << ret;
        return false;
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(pCodecContext, pPacket);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            kLogger.warning() << "avcodec_receive_packet failed:" << ret;
            return false;
        }
        av_packet_rescale_ts(pPacket, pCodecContext->time_base, pStream->time_base);
        pPacket->stream_index = pStream->index;
        ret = av_interleaved_write_frame(pFormatContext, pPacket);
        av_packet_unref(pPacket);
        if (ret < 0) {
            kLogger.warning() << "av_interleaved_write_frame failed:" << ret;
            return false;
        }
    }
    return true;
}

bool encodeStream(AVFormatContext* pFormatContext,
        EncoderStream* pEncoderStream,
        const std::vector<CSAMPLE>& interleavedStereo,
        AVPacket* pPacket) {
    AVCodecContext* pCodecContext = pEncoderStream->pCodecContext;
    AVFrame* pFrame = pEncoderStream->pFrame;
    const int frameSize = pCodecContext->frame_size;
    const qint64 numFrames =
            static_cast<qint64>(interleavedStereo.size() / kNumChannels);

    for (qint64 offset = 0; offset < numFrames; offset += frameSize) {
        const int currentFrameSize =
                static_cast<int>(std::min<qint64>(frameSize, numFrames - offset));

        if (av_frame_make_writable(pFrame) < 0) {
            kLogger.warning() << "av_frame_make_writable failed";
            return false;
        }
        pFrame->nb_samples = currentFrameSize;

        for (int channel = 0; channel < kNumChannels; ++channel) {
            auto* pChannelData = reinterpret_cast<int32_t*>(pFrame->data[channel]);
            for (int frame = 0; frame < currentFrameSize; ++frame) {
                const CSAMPLE sample = interleavedStereo[static_cast<size_t>(
                                                                  offset + frame) *
                                kNumChannels +
                        channel];
                const CSAMPLE clamped = std::clamp(sample, -1.0f, 1.0f);
                pChannelData[frame] = static_cast<int32_t>(clamped * 2147483647.0f);
            }
        }

        pFrame->pts = pEncoderStream->nextPts;
        pEncoderStream->nextPts += currentFrameSize;

        if (!sendAndWritePacket(pFormatContext,
                    pCodecContext,
                    pEncoderStream->pStream,
                    pFrame,
                    pPacket)) {
            return false;
        }
    }
    return true;
}

// Serializes the STEM manifest that StemInfoImporter expects to find in
// `moov/udta/stem`, describing the 4 stem streams that follow the premix.
QByteArray buildStemManifest() {
    QJsonArray stems;
    for (int i = 0; i < kNumStemSeparationTargets; ++i) {
        const auto target = static_cast<StemSeparationTarget>(i);
        QJsonObject stem;
        stem.insert(QStringLiteral("name"), stemSeparationTargetLabel(target));
        stem.insert(QStringLiteral("color"),
                stemSeparationTargetColor(target).name(QColor::HexRgb));
        stems.append(stem);
    }
    QJsonObject manifest;
    // Version of the Native Instruments STEM manifest format that
    // StemInfoImporter implements.
    manifest.insert(QStringLiteral("version"), 1);
    manifest.insert(QStringLiteral("stems"), stems);
    return QJsonDocument(manifest).toJson(QJsonDocument::Compact);
}

// Overwrites the 32 bit big endian box size at `offset` in `buffer`.
void setBoxSize(QByteArray* pBuffer, qint64 offset, quint32 size) {
    const quint32 bigEndianSize = qToBigEndian(size);
    std::memcpy(pBuffer->data() + offset, &bigEndianSize, sizeof(bigEndianSize));
}

QByteArray makeAtom(const char* type, const QByteArray& payload) {
    QByteArray atom;
    const quint32 bigEndianSize =
            qToBigEndian(static_cast<quint32>(payload.size() + 8));
    atom.append(reinterpret_cast<const char*>(&bigEndianSize), sizeof(bigEndianSize));
    atom.append(type, 4);
    atom.append(payload);
    return atom;
}

// Adds the STEM manifest atom to an MP4 file that FFmpeg has finished
// writing.
//
// Mixxx only treats an MP4 file as a STEM file if it carries this atom
// (SoundSource::getTypeFromFile() -> StemInfoImporter::hasStemAtom()), and
// libavformat has no API for writing arbitrary atoms, so it is spliced into
// the finished file here. Only `moov` is rewritten, and only when it is the
// last box in the file — growing it would otherwise invalidate the sample
// offsets stored in it.
bool addStemManifestAtom(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadWrite)) {
        kLogger.warning() << "Failed to reopen" << filePath << "to add the STEM manifest";
        return false;
    }

    qint64 moovOffset = -1;
    quint32 moovSize = 0;
    qint64 offset = 0;
    const qint64 fileSize = file.size();
    while (offset + 8 <= fileSize) {
        if (!file.seek(offset)) {
            break;
        }
        QByteArray header = file.read(8);
        if (header.size() != 8) {
            break;
        }
        const quint32 boxSize = qFromBigEndian<quint32>(
                reinterpret_cast<const uchar*>(header.constData()));
        const QByteArray boxType = header.mid(4, 4);
        if (boxSize < 8) {
            // 0 means "extends to end of file", 1 means a 64 bit size
            // follows; neither is handled here, and neither is produced by
            // FFmpeg's MP4 muxer for `moov`.
            kLogger.warning() << "Unsupported box size" << boxSize << "for box"
                              << boxType << "in" << filePath;
            return false;
        }
        if (boxType == QByteArrayLiteral("moov")) {
            moovOffset = offset;
            moovSize = boxSize;
        }
        offset += boxSize;
    }

    if (moovOffset < 0) {
        kLogger.warning() << "No moov box found in" << filePath;
        return false;
    }
    if (moovOffset + moovSize != fileSize) {
        kLogger.warning() << "moov is not the last box in" << filePath
                          << "- refusing to add the STEM manifest";
        return false;
    }

    if (!file.seek(moovOffset)) {
        return false;
    }
    QByteArray moov = file.read(moovSize);
    if (static_cast<quint32>(moov.size()) != moovSize) {
        return false;
    }

    const QByteArray stemAtom = makeAtom("stem", buildStemManifest());

    // Find an existing `udta` child of `moov` to put the manifest in, or
    // append a new one.
    qint64 udtaOffset = -1;
    quint32 udtaSize = 0;
    qint64 childOffset = 8;
    while (childOffset + 8 <= moov.size()) {
        const quint32 boxSize = qFromBigEndian<quint32>(
                reinterpret_cast<const uchar*>(moov.constData() + childOffset));
        if (boxSize < 8 || childOffset + boxSize > static_cast<quint32>(moov.size())) {
            break;
        }
        if (moov.mid(childOffset + 4, 4) == QByteArrayLiteral("udta")) {
            udtaOffset = childOffset;
            udtaSize = boxSize;
            break;
        }
        childOffset += boxSize;
    }

    if (udtaOffset >= 0) {
        moov.insert(udtaOffset + udtaSize, stemAtom);
        setBoxSize(&moov, udtaOffset, udtaSize + static_cast<quint32>(stemAtom.size()));
    } else {
        moov.append(makeAtom("udta", stemAtom));
    }

    setBoxSize(&moov, 0, static_cast<quint32>(moov.size()));

    if (!file.seek(moovOffset) || file.write(moov) != moov.size()) {
        kLogger.warning() << "Failed to write the STEM manifest into" << filePath;
        return false;
    }
    return file.resize(moovOffset + moov.size());
}

bool flushStream(AVFormatContext* pFormatContext,
        EncoderStream* pEncoderStream,
        AVPacket* pPacket) {
    return sendAndWritePacket(pFormatContext,
            pEncoderStream->pCodecContext,
            pEncoderStream->pStream,
            nullptr,
            pPacket);
}

} // namespace

bool writeStemContainer(
        const QString& filePath,
        int sampleRate,
        const std::vector<std::vector<CSAMPLE>>& streams) {
    const size_t expectedStreamCount = 1 + kNumStemSeparationTargets;
    if (streams.size() != expectedStreamCount) {
        kLogger.warning() << "Expected" << expectedStreamCount
                           << "audio streams (premix +"
                           << kNumStemSeparationTargets << "stems), got"
                           << streams.size();
        return false;
    }

    const AVCodec* pCodec = avcodec_find_encoder(AV_CODEC_ID_ALAC);
    if (!pCodec) {
        kLogger.warning() << "ALAC encoder not available in this FFmpeg build";
        return false;
    }

    AVFormatContext* pFormatContext = nullptr;
    const QByteArray filePathUtf8 = filePath.toUtf8();
    if (avformat_alloc_output_context2(
                &pFormatContext, nullptr, nullptr, filePathUtf8.constData()) < 0 ||
            !pFormatContext) {
        kLogger.warning() << "Failed to determine output container format for"
                           << filePath;
        return false;
    }

    std::vector<EncoderStream> encoderStreams(streams.size());
    bool ok = true;

    for (size_t i = 0; ok && i < streams.size(); ++i) {
        EncoderStream& encoderStream = encoderStreams[i];

        encoderStream.pStream = avformat_new_stream(pFormatContext, nullptr);
        encoderStream.pCodecContext = avcodec_alloc_context3(pCodec);
        if (!encoderStream.pStream || !encoderStream.pCodecContext) {
            ok = false;
            break;
        }

        encoderStream.pCodecContext->sample_rate = sampleRate;
        encoderStream.pCodecContext->sample_fmt = AV_SAMPLE_FMT_S32P;
        encoderStream.pCodecContext->time_base = AVRational{1, sampleRate};
        setStereoChannelLayout(encoderStream.pCodecContext);

        if (pFormatContext->oformat->flags & AVFMT_GLOBALHEADER) {
            encoderStream.pCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        if (avcodec_open2(encoderStream.pCodecContext, pCodec, nullptr) < 0) {
            kLogger.warning() << "avcodec_open2 failed for stream" << i;
            ok = false;
            break;
        }
        if (avcodec_parameters_from_context(
                    encoderStream.pStream->codecpar, encoderStream.pCodecContext) < 0) {
            ok = false;
            break;
        }
        encoderStream.pStream->time_base = encoderStream.pCodecContext->time_base;

        encoderStream.pFrame = av_frame_alloc();
        encoderStream.pFrame->format = encoderStream.pCodecContext->sample_fmt;
        encoderStream.pFrame->sample_rate = sampleRate;
        encoderStream.pFrame->nb_samples = encoderStream.pCodecContext->frame_size;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100) // FFmpeg 5.1
        av_channel_layout_copy(&encoderStream.pFrame->ch_layout,
                &encoderStream.pCodecContext->ch_layout);
#else
        encoderStream.pFrame->channels = kNumChannels;
        encoderStream.pFrame->channel_layout = AV_CH_LAYOUT_STEREO;
#endif
        if (av_frame_get_buffer(encoderStream.pFrame, 0) < 0) {
            ok = false;
            break;
        }
    }

    if (ok && !(pFormatContext->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&pFormatContext->pb, filePathUtf8.constData(), AVIO_FLAG_WRITE) < 0) {
            kLogger.warning() << "Failed to open" << filePath << "for writing";
            ok = false;
        }
    }

    if (ok && avformat_write_header(pFormatContext, nullptr) < 0) {
        kLogger.warning() << "avformat_write_header failed for" << filePath;
        ok = false;
    }

    if (ok) {
        AVPacket* pPacket = av_packet_alloc();
        for (size_t i = 0; ok && i < streams.size(); ++i) {
            ok = encodeStream(pFormatContext, &encoderStreams[i], streams[i], pPacket);
        }
        for (size_t i = 0; ok && i < encoderStreams.size(); ++i) {
            ok = flushStream(pFormatContext, &encoderStreams[i], pPacket);
        }
        av_packet_free(&pPacket);

        if (ok) {
            ok = av_write_trailer(pFormatContext) == 0;
        }
    }

    if (!(pFormatContext->oformat->flags & AVFMT_NOFILE) && pFormatContext->pb) {
        avio_closep(&pFormatContext->pb);
    }
    for (EncoderStream& encoderStream : encoderStreams) {
        if (encoderStream.pFrame) {
            av_frame_free(&encoderStream.pFrame);
        }
        if (encoderStream.pCodecContext) {
            avcodec_free_context(&encoderStream.pCodecContext);
        }
    }
    avformat_free_context(pFormatContext);

    if (ok) {
        // Must happen after the container is closed and complete.
        ok = addStemManifestAtom(filePath);
    }

    if (!ok) {
        kLogger.warning() << "Failed to write STEM container" << filePath;
    }
    return ok;
}

} // namespace mixxx
