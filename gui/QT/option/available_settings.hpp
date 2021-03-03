#ifndef AVAILABLE_SETTINGS_HPP
#define AVAILABLE_SETTINGS_HPP

#include <string>
#include <vector>
#include <QStringList>

enum SettingType{
	VIDEO_SRC = 0,
	VIDEO_DISPLAY,
	VIDEO_COMPRESS,
	VIDEO_CAPTURE_FILTER,
	AUDIO_SRC,
	AUDIO_PLAYBACK,
	AUDIO_COMPRESS,
	SETTING_TYPE_UNKNOWN,
	SETTING_TYPE_COUNT
};

struct SettingVal{
	std::string opt;
	std::string val;
};

struct DeviceMode{
	std::string name;
	std::vector<SettingVal> opts;
};

struct Device{
	std::string name;
	std::string type;
	std::string deviceOpt;

	std::vector<DeviceMode> modes;
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
	void queryDevice(const QString &line, size_t offset);
	void queryVideoCompress(const QString &line, size_t offset);

	void queryAll(const std::string &executable);

	bool isAvailable(const std::string &name, SettingType type) const;
	std::vector<std::string> getAvailableSettings(SettingType type) const;

	const std::vector<Device>& getDevices(SettingType type) const;

	const std::vector<CompressModule>& getVideoCompressModules() const {
		return videoCompressModules;
	}

	const std::vector<Codec>& getVideoCompressCodecs() const {
		return videoCompressCodecs;
	}

private:

	void queryProcessLine(const QString& line, bool *endMarker = nullptr);

	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Device> devices[SETTING_TYPE_COUNT];
	std::vector<CompressModule> videoCompressModules;
	std::vector<Codec> videoCompressCodecs;

};

#endif
