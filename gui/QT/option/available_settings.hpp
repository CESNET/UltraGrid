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

struct CaptureMode{
	std::string name;
	std::vector<SettingVal> opts;
};

struct Capturer{
	std::string name;
	std::string type;
	std::string deviceOpt;

	std::vector<CaptureMode> modes;
};

class AvailableSettings{
public:
	void query(const std::string &executable, SettingType type);
	void queryCap(const QStringList &lines, SettingType type, const char *capStr);
	void queryCapturers(const QStringList &lines);

	void queryAll(const std::string &executable);

	bool isAvailable(const std::string &name, SettingType type) const;
	std::vector<std::string> getAvailableSettings(SettingType type) const;

	std::vector<Capturer> getCapturers() const;

private:
	std::vector<std::string> available[SETTING_TYPE_COUNT];
	std::vector<Capturer> capturers;

};

#endif
