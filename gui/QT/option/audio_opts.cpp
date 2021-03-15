#include <cstring>

#include "ui_ultragrid_window.h"

#include "audio_opts.hpp"
#include "combobox_ui.hpp"

std::vector<SettingItem> getAudioCompress(AvailableSettings *availSettings){
	const std::string optStr = "audio.compress";
	std::vector<SettingItem> res;

	SettingItem defaultItem;
	defaultItem.name = "None";
	defaultItem.opts.push_back({optStr, ""});
	res.push_back(std::move(defaultItem));

	for(const auto &i : availSettings->getAvailableSettings(AUDIO_COMPRESS)){
		SettingItem item;
		item.name = i;
		item.opts.push_back({optStr, i});
		res.push_back(std::move(item));
	}

	return res;
}

std::vector<SettingItem> getAudioPlayback(AvailableSettings *availSettings){
	const std::string optStr = "audio.playback";
	std::vector<SettingItem> res;

	SettingItem defaultItem;
	defaultItem.name = "None";
	defaultItem.opts.push_back({optStr, ""});
	res.push_back(std::move(defaultItem));

	for(const auto &i : availSettings->getDevices(AUDIO_PLAYBACK)){
		SettingItem item;
		item.name = i.name;
		item.opts.push_back({optStr, i.type});
		item.opts.push_back({optStr + "." + i.type + ".device", i.deviceOpt});
		auto embedAudioIt = i.extra.find("isEmbeddedAudio");
		if(embedAudioIt != i.extra.end() && embedAudioIt->second == "t"){
			item.conditions.push_back({{{"video.display.embeddedAudioAvailable", "t"}, false}});
		}
		res.push_back(std::move(item));
	}

	return res;
}

std::vector<SettingItem> getAudioSrc(AvailableSettings *availSettings){
	const std::string optStr = "audio.source";
	std::vector<SettingItem> res;

	SettingItem defaultItem;
	defaultItem.name = "None";
	defaultItem.opts.push_back({optStr, ""});
	res.push_back(std::move(defaultItem));

	for(const auto &i : availSettings->getDevices(AUDIO_SRC)){
		SettingItem item;
		item.name = i.name;
		item.opts.push_back({optStr, i.type});
		item.opts.push_back({optStr + "." + i.type + ".device", i.deviceOpt});
		auto embedAudioIt = i.extra.find("isEmbeddedAudio");
		if(embedAudioIt != i.extra.end() && embedAudioIt->second == "t"){
			item.conditions.push_back({{{"video.source.embeddedAudioAvailable", "t"}, false}});
		}
		res.push_back(std::move(item));
	}

	return res;
}

void audioCompressionCallback(Option &opt, bool suboption, void *opaque){
	Ui::UltragridWindow *win = static_cast<Ui::UltragridWindow *>(opaque);

	if(suboption)
		return;

	const char *const losslessCodecs[] = {
		"",
		"FLAC",
		"u-law",
		"A-law",
		"PCM"
	};

	bool enableBitrate = true;

	for(const auto str : losslessCodecs){
		if(std::strcmp(str, opt.getValue().c_str()) == 0){
			enableBitrate = false;
			break;
		}
	}

	opt.getSettings()->getOption("audio.compress.bitrate").setEnabled(enableBitrate);
	win->audioBitrateEdit->setEnabled(enableBitrate);
	win->audioBitrateLabel->setEnabled(enableBitrate);
}

