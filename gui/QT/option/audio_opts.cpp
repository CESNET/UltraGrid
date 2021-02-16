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

const char * const sdiAudioCards[] = {
    "decklink",
    "aja",
    "dvs",
    "deltacast",
    "ndi",
};

const char * const sdiAudio[] = {
    "analog",
    "AESEBU",
    "embedded",
};

static std::vector<std::vector<ConditionItem>> getSdiCond(const std::string &opt){
    std::vector<std::vector<ConditionItem>> res;

    std::vector<ConditionItem> clause;

    for(const auto &i : sdiAudioCards){
        clause.push_back({{opt, i}, false});
    }

    res.push_back(std::move(clause));

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
        for(const auto &sdi : sdiAudio){
            if(std::strcmp(sdi, i.type.c_str()) == 0){
                item.conditions = getSdiCond("video.display");
                break;
            }
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
        for(const auto &sdi : sdiAudio){
            if(std::strcmp(sdi, i.type.c_str()) == 0){
                item.conditions = getSdiCond("video.source");
                break;
            }
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

