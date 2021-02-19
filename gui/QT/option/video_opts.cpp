#include <cstring>
#include <QLabel>

#include "video_opts.hpp"
#include "combobox_ui.hpp"
#include "lineedit_ui.hpp"

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings){
	const char * const whiteList[] = {
		"bitflow",
	};
    const std::string optStr = "video.source";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getDevices(VIDEO_SRC)){
        SettingItem item;
        item.name = i.name;
        item.opts.push_back({optStr, i.type});
        item.opts.push_back({optStr + "." + i.type + ".device", i.deviceOpt});
        res.push_back(std::move(item));
    }

    for(const auto &i : availSettings->getAvailableSettings(VIDEO_SRC)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        bool whiteListed = false;
        for(const auto &white : whiteList){
            if(std::strcmp(white, i.c_str()) == 0){
                whiteListed = true;
            }
        }
        if(!whiteListed){
            item.conditions.push_back({{{"advanced", "t"}, false}});
        }
        res.push_back(std::move(item));
    }

    return res;
}

std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    for(const auto &cap : availSettings->getDevices(VIDEO_SRC)){
        for(const auto &mode : cap.modes){
            SettingItem item;
            item.name = mode.name;
            item.conditions.push_back({{{"video.source", cap.type}, false}});
            for(const auto &opt : mode.opts){
                item.opts.push_back(
                        {"video.source." + cap.type + "." + opt.opt, opt.val});
            }
            res.push_back(std::move(item));
        } 
    }

    return res;
}

std::vector<SettingItem> getVideoDisplay(AvailableSettings *availSettings){
	const char * const whiteList[] = {
		"dvs",
		"quicktime"
	};

    const std::string optStr = "video.display";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

    for(const auto &i : availSettings->getDevices(VIDEO_DISPLAY)){
        SettingItem item;
        item.name = i.name;
        item.opts.push_back({optStr, i.type});
        res.push_back(std::move(item));
    }

    for(const auto &i : availSettings->getAvailableSettings(VIDEO_DISPLAY)){
        SettingItem item;
        item.name = i;
        item.opts.push_back({optStr, i});
        bool whiteListed = false;
        for(const auto &white : whiteList){
            if(std::strcmp(white, i.c_str()) == 0){
                whiteListed = true;
            }
        }
        if(!whiteListed){
            item.conditions.push_back({{{"advanced", "t"}, false}});
        }
        res.push_back(std::move(item));
    }

    return res;
}

struct VideoCompressItem{
	const char * displayName;
	const char * value;
	bool isLibav;
};

static const VideoCompressItem videoCodecs[] = {
	{"None", "", false},
	{"H.264", "H.264", true},
	{"H.265", "H.265", true},
	{"MJPEG", "MJPEG", true},
	{"VP8", "VP8", true},
	{"Jpeg", "jpeg", false},
	{"Cineform", "cineform", false},
};

void populateVideoCompressSettings(AvailableSettings *availSettings,
		Settings* settings)
{
	for(const auto& mod : availSettings->getVideoCompressModules()){
		std::string codecOptKey = mod.name + ".codec";
		settings->addOption(codecOptKey,
				Option::SilentOpt,
				"",
				mod.codecs[0].name,
				false,
				"video.compress",
				mod.name);

		for(const auto& modOption: mod.opts){
			settings->addOption(mod.name + "." + modOption.key,
					modOption.booleanOpt ? Option::BoolOpt : Option::StringOpt,
					modOption.optStr,
					"",
					false,
					"video.compress",
					mod.name
					);
		}

		for(const auto& codec : mod.codecs){
			std::string optName = "video.compress.";
			optName += mod.name;
			optName += ".codec";

			settings->addOption(codec.name + ".encoder",
					Option::StringOpt,
					"",
					codec.encoders[0].optStr,
					true,
					optName,
					codec.name);
		}
	}
}

std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    const std::string optStr = "video.compress";

	for(const auto& mod : availSettings->getVideoCompressModules()){
		for(const auto& codec : mod.codecs){
			SettingItem item;
			item.name = codec.name;
			item.opts.push_back({"video.compress", mod.name});
			item.opts.push_back({"video.compress." + mod.name + ".codec", codec.name});

			res.push_back(std::move(item));
		}
	}

    return res;
}

void videoCompressBitrateCallback(Option &opt, bool suboption, void *opaque){
	VideoBitrateCallbackData *data = static_cast<VideoBitrateCallbackData *>(opaque);
	if(suboption)
		return;

	Settings *settings = opt.getSettings();
	std::string mod = settings->getOption("video.compress").getValue();
	std::string codec = settings->getOption("video.compress." + mod + ".codec").getValue();

	bool enableEdit = false;
	QString toolTip;
	QString qualityLabel;

	for(const auto& compMod : data->availSettings->getVideoCompressModules()){
		if(compMod.name == mod){
			for(const auto& modOpt : compMod.opts){
				if(modOpt.key == "quality"){
					enableEdit = true;
					toolTip = QString::fromStdString(modOpt.displayDesc);
					qualityLabel = QString::fromStdString(modOpt.displayName);
					break;
				}
			}
		}
	}

    data->lineEditUi->setOpt("video.compress." + mod + ".quality");
	data->lineEditUi->setEnabled(enableEdit);
	data->lineEditUi->setToolTip(toolTip);
	if(data->label){
		data->label->setEnabled(enableEdit);
		data->label->setToolTip(toolTip);
		if(!qualityLabel.isEmpty())
			data->label->setText(qualityLabel);
	}
}
