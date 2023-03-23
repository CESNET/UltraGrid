#ifndef AVAILABLE_SETTINGS_HPP
#define AVAILABLE_SETTINGS_HPP

#include <string>
#include <string_view>
#include <vector>
#include <map>

#define SETTING_TYPE_LIST \
	X(VIDEO_SRC, "capture") \
	X(VIDEO_DISPLAY, "display") \
	X(VIDEO_COMPRESS, "compress") \
	X(VIDEO_CAPTURE_FILTER, "capture_filter") \
	X(AUDIO_SRC, "audio_cap") \
	X(AUDIO_PLAYBACK, "audio_play") \
	X(AUDIO_COMPRESS, "audio_compress") \
	X(AUDIO_FILTER, "audio_filter") \
	X(SETTING_TYPE_UNKNOWN, "unknown")


#define X(type, str) type,
enum SettingType{
	SETTING_TYPE_LIST
	SETTING_TYPE_COUNT
};
#undef X

const char *settingTypeToStr(SettingType type);
SettingType strToSettingType(std::string_view sv);

struct SettingVal{
	std::string opt;
	std::string val;
};

struct DeviceMode{
	std::string name;
	std::vector<SettingVal> opts;
};

struct CapabOpt{
	std::string displayName; //Name displayed to user
	std::string displayDesc; //Description displayed to user

	/* internal name of option, options that are used in the same way should
	 * have the same key (e.g. both bitrate for libavcodec and quality for jpeg
	 * should have the same key). This allows using different labels displayed
	 * to user */
	std::string key;
	std::string optStr; //Option string to pass to ug (e.g. ":bitrate=")

	bool booleanOpt; //If true, GUI shows a checkbox instead of text edit
};

struct Device{
	std::string name;
	std::string type; //UltraGrid module name (e.g. v4l2)
	std::string deviceOpt; //Specific device (e.g. ":device=/dev/video0")
	SettingType settingType = SETTING_TYPE_UNKNOWN;

	std::map<std::string, std::string> extra;

	std::vector<DeviceMode> modes;
	std::vector<CapabOpt> opts;
};

struct Encoder{
	std::string name;
	std::string optStr;
};

struct Codec{
	std::string name;
	std::string module_name;
	std::vector<Encoder> encoders;
	int priority;
};

struct CompressModule{
	std::string name;
	std::vector<CapabOpt> opts;
};

class AvailableSettings{
public:
	void queryBegin();
	void queryBeginPass();
	void queryLine(std::string_view line);
	void queryEnd();

	bool isAvailable(const std::string &name, SettingType type) const;
	const std::vector<std::string>& getAvailableSettings(SettingType type) const;

	const std::vector<Device>& getDevices(SettingType type) const;

	const std::vector<CompressModule>& getVideoCompressModules() const {
		return videoCompressModules;
	}

	const std::vector<Codec>& getVideoCompressCodecs() const {
		return videoCompressCodecs;
	}

private:

	void queryProcessLine(std::string_view line);
	void queryDevice(std::string_view line);
	void queryVideoCompress(std::string_view line);

	void parseStateVersion(std::string_view line);

	void (AvailableSettings::*parseFunc)(std::string_view str) = nullptr;

	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Device> devices[SETTING_TYPE_COUNT];
	std::vector<CompressModule> videoCompressModules;
	std::vector<Codec> videoCompressCodecs;

	bool endMarkerFound = false;

};

#endif
