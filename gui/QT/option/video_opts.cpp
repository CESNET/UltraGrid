#include <cstring>

#include "video_opts.hpp"
#include "combobox_ui.hpp"
#include "lineedit_ui.hpp"

std::vector<SettingItem> getVideoSrc(AvailableSettings *availSettings){
	const char * const whiteList[] = {
		"aja",
		"dvs"
	};
    const std::string optStr = "video.source";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

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

    for(const auto &i : availSettings->getCapturers()){
        SettingItem item;
        item.name = i.name;
        item.opts.push_back({optStr, i.type});
        item.opts.push_back({optStr + "." + i.type + ".device", i.deviceOpt});
        res.push_back(std::move(item));
    }

    return res;
}

std::vector<SettingItem> getVideoModes(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    for(const auto &cap : availSettings->getCapturers()){
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
		"gl",
		"sdl",
		"decklink",
		"aja",
		"dvs"
	};

    const std::string optStr = "video.display";

    std::vector<SettingItem> res;

    SettingItem defaultItem;
    defaultItem.name = "None";
    defaultItem.opts.push_back({optStr, ""});
    res.push_back(std::move(defaultItem));

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

static bool isLibavCodec(const std::string &str){
	for(const auto &i : videoCodecs){
		if(i.value == str && i.isLibav)
			return true;
	}

	return false;
}

std::vector<SettingItem> getVideoCompress(AvailableSettings *availSettings){
    std::vector<SettingItem> res;

    const std::string optStr = "video.compress";

    for(const auto &i : videoCodecs){
        SettingItem item;
        item.name = i.displayName;
        std::string value;
        if(i.isLibav){
            value = "libavcodec";
            item.opts.push_back({optStr + ".libavcodec.codec", i.value});
        } else {
            value = i.value;
        }
        item.opts.push_back({optStr, value});

        bool available = value.empty() ? true : false;
        for(const auto &i : availSettings->getAvailableSettings(VIDEO_COMPRESS)){
            if(value == i){
                available = true;
                break;
            }
        }
        if(!available){
            item.conditions.push_back({{{"advanced", "t"}, false}});
        }

        res.push_back(std::move(item));
    }

    return res;
}

static std::string getBitrateOpt(Settings *settings){
	std::string codec = settings->getOption("video.compress").getValue();
	std::string opt = "video.compress." + codec;
	if(codec == "libavcodec"){
		opt += ".codec."
			+ settings->getOption("video.compress.libavcodec.codec").getValue()
			+ ".bitrate";
	} else if (codec == "jpeg") {
		opt += ".quality";
	} else {
		opt += codec + ".bitrate";
	}

	return opt;
}

void videoCompressBitrateCallback(LineEditUi *bitrateLine, Option &opt, bool suboption){
    if(suboption)
        return;

    bitrateLine->setOpt(getBitrateOpt(opt.getSettings()));
}
