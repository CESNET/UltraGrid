#ifndef VIDEO_OPTS_HPP
#define VIDEO_OPTS_HPP

#include <vector>
#include "available_settings.hpp"
#include "settings.hpp"

struct SettingItem;

class LineEditUi;

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoDisplay(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings);

void populateVideoCompressSettings(AvailableSettings *availSettings,
		Settings* settings);

void videoCompressBitrateCallback(Option &opt, bool suboption, void *opaque);


#endif
