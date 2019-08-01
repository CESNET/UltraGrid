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

class AvailableSettings{
public:
	void queryCap(const QStringList &lines, SettingType type, const char *capStr);
	void queryDevices(const QStringList &lines);

	void queryAll(const std::string &executable);

	bool isAvailable(const std::string &name, SettingType type) const;
	std::vector<std::string> getAvailableSettings(SettingType type) const;

	const std::vector<Device>& getDevices(SettingType type) const;

private:
	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Device> devices[SETTING_TYPE_COUNT];

};

#endif
