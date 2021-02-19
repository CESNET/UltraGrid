#ifndef VIDEO_OPTS_HPP
#define VIDEO_OPTS_HPP

#include <vector>
#include "available_settings.hpp"
#include "settings.hpp"
#include "extra_callback_data.hpp"

struct SettingItem;

class LineEditUi;

class QLabel;
struct VideoBitrateCallbackData : public ExtraCallbackData{
	const AvailableSettings *availSettings;
	LineEditUi *lineEditUi;
	QLabel *label;
};

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoDisplay(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings);
std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings);

void populateVideoCompressSettings(AvailableSettings *availSettings,
		Settings* settings);

void videoCompressBitrateCallback(Option &opt, bool suboption, void *opaque);


#endif
