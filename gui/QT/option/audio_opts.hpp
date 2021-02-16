#ifndef AUDIO_OPTS_HPP
#define AUDIO_OPTS_HPP

#include <vector>
#include "available_settings.hpp"
#include "settings.hpp"

struct SettingItem;

namespace Ui{
    class UltragridWindow;
};

std::vector<SettingItem> getAudioSrc(AvailableSettings *availSettings);
std::vector<SettingItem> getAudioPlayback(AvailableSettings *availSettings);
std::vector<SettingItem> getAudioCompress(AvailableSettings *availSettings);

void audioCompressionCallback(Option &opt, bool suboption, void *);

#endif
