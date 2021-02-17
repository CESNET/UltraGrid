#include <QProcess>
#include <QString>
#include <QStringList>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <cstring>
#include "available_settings.hpp"

#include <iostream>
#include <map>

static bool vectorContains(const std::vector<std::string> &v, const std::string & s){
	for(unsigned i = 0; i < v.size(); i++){
		if(v[i] == s)
			return true;
	}
	return false;
}

static QStringList getProcessOutput(const std::string& executable, const QStringList& args){
	QProcess process;
	process.start(executable.c_str(), args);
	process.waitForFinished();

	QString output = QString(process.readAllStandardOutput());
	QStringList lines = output.split(QRegularExpression("\n|\r\n|\r"));

	return lines;
}

void AvailableSettings::queryAll(const std::string &executable){
	QStringList lines = getProcessOutput(executable, QStringList() << "--capabilities");

	queryCap(lines, VIDEO_SRC, "[cap][capture] ");
	queryCap(lines, VIDEO_DISPLAY, "[cap][display] ");
	queryCap(lines, VIDEO_COMPRESS, "[cap][compress] ");
	queryCap(lines, VIDEO_CAPTURE_FILTER, "[cap][capture_filter] ");
	queryCap(lines, AUDIO_SRC, "[cap][audio_cap] ");
	queryCap(lines, AUDIO_PLAYBACK, "[cap][audio_play] ");
	queryCap(lines, AUDIO_COMPRESS, "[cap][audio_compress] ");
	
	queryDevices(lines);
	queryVideoCompress(lines);
}

void AvailableSettings::queryVideoCompress(const QStringList &lines){
	const char * const devStr = "[capability][video_compress][v2]";
	size_t devStrLen = strlen(devStr);

	videoCompressModules.clear();

	videoCompressModules.push_back({
			"libavcodec",
			{
				{"Bitrate", "Video Bitrate desc", "quality", ":bitrate=", false},
				{"Crf", "Constant rate factor", "crf", ":crf=", false},
			},
			{
				{
					"H.264",
					{
						{"default", ":codec=H.264"},
						{"libx264", ":encoder=libx264"}
					}
				}
			}
			});

	videoCompressModules.push_back({
			"cineform",
			{
				{"Quality", "Video quality desc", "quality", ":quality=", false},
				{"Threads", "Number of threads to use", "threads", ":threads=", false},
			},
			{
				{
					"Cineform",
					{
						{"default", ""}
					}
				}
			}
			});
}

void AvailableSettings::queryDevices(const QStringList &lines){
	const char * const devStr = "[capability][device][v2]";
	size_t devStrLen = strlen(devStr);

	for(auto& i : devices){
		i.clear();
	}

	static std::map<std::string, SettingType> settingTypeMap = {
		{"video_cap", VIDEO_SRC},
		{"video_disp", VIDEO_DISPLAY},
		{"audio_play", AUDIO_PLAYBACK},
		{"audio_cap", AUDIO_SRC},
	};

	foreach ( const QString &line, lines ) {
		if(line.startsWith(devStr)){
			QJsonDocument doc = QJsonDocument::fromJson(line.mid(devStrLen).toUtf8());
			if(!doc.isObject())
				return;

			Device dev;
			QJsonObject obj = doc.object();
			if(obj.contains("name") && obj["name"].isString()){
				dev.name = obj["name"].toString().toStdString();
			}
			if(obj.contains("type") && obj["type"].isString()){
				dev.type = obj["type"].toString().toStdString();
			}
			if(obj.contains("device") && obj["device"].isString()){
				dev.deviceOpt = obj["device"].toString().toStdString();
			}

			if(obj.contains("modes") && obj["modes"].isArray()){
				for(const QJsonValue &val : obj["modes"].toArray()){
					if(val.isObject()){
						QJsonObject modeJson = val.toObject();

						DeviceMode mode;

						if(modeJson.contains("name") && modeJson["name"].isString()){
							mode.name = modeJson["name"].toString().toStdString();
						}

						if(modeJson.contains("opts") && modeJson["opts"].isObject()){
							QJsonObject modeOpts = modeJson["opts"].toObject();
							for(const QString &key : modeOpts.keys()){
								if(modeOpts[key].isString()){
									mode.opts.push_back(SettingVal{key.toStdString(),
											modeOpts[key].toString().toStdString()});
								}
							}
							dev.modes.push_back(std::move(mode));
						}
					}
				}
			}

			SettingType settingType = SETTING_TYPE_UNKNOWN;
			if(obj.contains("purpose") && obj["purpose"].isString()){
				settingType = settingTypeMap[obj["purpose"].toString().toStdString()];
			}
			devices[settingType].push_back(std::move(dev));
		}
	}
}

void AvailableSettings::queryCap(const QStringList &lines,
		SettingType type,
		const char *capStr)
{
	available[type].clear();

	size_t capStrLen = strlen(capStr);
	foreach ( const QString &line, lines ) {
		if(line.startsWith(capStr)){
			available[type].push_back(line.mid(capStrLen).toStdString());
		}
	}
}

bool AvailableSettings::isAvailable(const std::string &name, SettingType type) const{
	return vectorContains(available[type], name);
}

std::vector<std::string> AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

const std::vector<Device>& AvailableSettings::getDevices(SettingType type) const{
	return devices[type];
}

