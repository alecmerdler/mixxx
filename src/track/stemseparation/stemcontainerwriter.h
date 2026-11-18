#pragma once

#include <QString>

#include <vector>

#include "util/types.h"

namespace mixxx {

// Writes a STEM-compatible container file (premix + 4 stem streams, all
// ALAC-encoded stereo at the same sample rate) that SoundSourceSTEM can open
// for playback through Mixxx's existing STEM engine/UI pipeline.
//
// `streams` must contain exactly 1 + kNumStemSeparationTargets entries, all
// of equal length: the premix (sum of all stems) first, followed by the
// individual stems in StemSeparationTarget order. Each entry contains
// interleaved stereo samples in [-1, 1].
bool writeStemContainer(
        const QString& filePath,
        int sampleRate,
        const std::vector<std::vector<CSAMPLE>>& streams);

} // namespace mixxx
