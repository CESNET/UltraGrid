#include <QProcess>
#include <QString>
#include <QStringList>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMessageBox>
#include <cstring>
#include "available_settings.hpp"

#include <iostream>
#include <map>
#include <algorithm>

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

struct {
	SettingType type;
	const char *capStr;
} modulesQueryInfo[] = {
	{VIDEO_SRC, "[cap][capture] "},
	{VIDEO_DISPLAY, "[cap][display] "}, 
	{VIDEO_COMPRESS, "[cap][compress] "},
	{VIDEO_CAPTURE_FILTER, "[cap][capture_filter] "},
	{AUDIO_SRC, "[cap][audio_cap] "},
	{AUDIO_PLAYBACK, "[cap][audio_play] "},
	{AUDIO_COMPRESS, "[cap][audio_compress] "},
};

void AvailableSettings::queryProcessLine(const QString& line, bool *endMarker){
	for(const auto& i : modulesQueryInfo){
		if(line.startsWith(i.capStr)){
			available[i.type].push_back(line.mid(strlen(i.capStr)).toStdString());
			return;
		}
	}

	const char * const devStr = "[capability][device]";
	const char * const codecStr = "[capability][video_compress]";
	const char * const endMarkerStr = "[capability][end]";

	if(line.startsWith(devStr)){
		queryDevice(line, strlen(devStr));
	} else if(line.startsWith(codecStr)){
		queryVideoCompress(line, strlen(codecStr));
	} else if(line.startsWith(endMarkerStr)){
		*endMarker = true;
	}
}

void AvailableSettings::queryAll(const std::string &executable){
	QStringList lines = getProcessOutput(executable, QStringList() << "--capabilities");

	for(const auto& i : modulesQueryInfo){
		available[i.type].clear();
	}
	for(auto& i : devices){
		i.clear();
	}
	videoCompressModules.clear();
	videoCompressCodecs.clear();
	videoCompressModules.emplace_back(CompressModule{"", {}, });
	videoCompressCodecs.emplace_back(Codec{"None", "", {Encoder{"default", ""}}, 0});

	int i;
	int ver = 0;
	const char *verStr = "[capability][start] version ";
	for(i = 0; i < lines.size(); i++){
		if(lines[i].startsWith(verStr)){
			ver = lines[i].mid(strlen(verStr)).toInt();
			break;
		}
	}

	const int expectedVersion = 3;
	if(ver != expectedVersion){
		QMessageBox msgBox;
		QString msg = "Capabilities start marker with expected version not found"
				" in ug output.";
		if(ver != 0)
			msg += " Version found: " + QString::number(ver);
		msgBox.setText(msg);
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.exec();
		return;
	}

	bool endMarkerFound = false;
	for(; i < lines.size(); i++) {
		queryProcessLine(lines[i], &endMarkerFound);

		if(endMarkerFound)
			break;
	}

	if(!endMarkerFound){
		QMessageBox msgBox;
		msgBox.setText("Capabilities end marker not found in ug output."
				" Some options may be missing as result.");
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.exec();
	}

	std::sort(videoCompressCodecs.begin(), videoCompressCodecs.end(),
			[](const Codec& a, const Codec& b){
				return a.priority < b.priority;
			});
}

static void maybeWriteString (const QJsonObject& obj,
		const char *key,
		std::string& result)
{
		if(obj.contains(key) && obj[key].isString()){
			result = obj[key].toString().toStdString();
		}
}

void AvailableSettings::queryVideoCompress(const QString &line, size_t offset){
	QJsonDocument doc = QJsonDocument::fromJson(line.mid(offset).toUtf8());
	if(!doc.isObject())
		return;

	CompressModule compMod;
	QJsonObject obj = doc.object();
	maybeWriteString(obj, "name", compMod.name);

	if(obj.contains("options") && obj["options"].isArray()){
		for(const QJsonValue &val : obj["options"].toArray()){
			QJsonObject optJson = val.toObject();

			CapabOpt capabOpt;
			maybeWriteString(optJson, "display_name", capabOpt.displayName);
			maybeWriteString(optJson, "display_desc", capabOpt.displayDesc);
			maybeWriteString(optJson, "key", capabOpt.key);
			maybeWriteString(optJson, "opt_str", capabOpt.optStr);
			if(optJson.contains("is_boolean") && optJson["is_boolean"].isString()){
				capabOpt.booleanOpt = optJson["is_boolean"].toString() == "t";
			}

			compMod.opts.emplace_back(std::move(capabOpt));
		}
	}

	if(obj.contains("codecs") && obj["codecs"].isArray()){
		for(const QJsonValue &val : obj["codecs"].toArray()){
			QJsonObject codecJson = val.toObject();

			Codec codec;
			maybeWriteString(codecJson, "name", codec.name);
			codec.module_name = compMod.name;
			if(codecJson.contains("priority") && codecJson["priority"].isDouble()){
				codec.priority = codecJson["priority"].toInt();
			}

			if(codecJson.contains("encoders") && codecJson["encoders"].isArray()){
				for(const QJsonValue &val : codecJson["encoders"].toArray()){
					QJsonObject encoderJson = val.toObject();

					Encoder encoder;
					maybeWriteString(encoderJson, "name", encoder.name);
					maybeWriteString(encoderJson, "opt_str", encoder.optStr);

					codec.encoders.emplace_back(std::move(encoder));
				}
			}

			if(codec.encoders.empty()){
				codec.encoders.emplace_back(Encoder{"default", ""});
			}

			videoCompressCodecs.emplace_back(std::move(codec));
		}
	}

	videoCompressModules.emplace_back(std::move(compMod));
}

void AvailableSettings::queryDevice(const QString &line, size_t offset){
	static std::map<std::string, SettingType> settingTypeMap = {
		{"video_cap", VIDEO_SRC},
		{"video_disp", VIDEO_DISPLAY},
		{"audio_play", AUDIO_PLAYBACK},
		{"audio_cap", AUDIO_SRC},
	};

	QJsonDocument doc = QJsonDocument::fromJson(line.mid(offset).toUtf8());
	if(!doc.isObject())
		return;

	Device dev;
	QJsonObject obj = doc.object();
	maybeWriteString(obj, "name", dev.name);
	maybeWriteString(obj, "type", dev.type);
	maybeWriteString(obj, "device", dev.deviceOpt);

	if(obj.contains("modes") && obj["modes"].isArray()){
		for(const QJsonValue &val : obj["modes"].toArray()){
			if(val.isObject()){
				QJsonObject modeJson = val.toObject();

				DeviceMode mode;

				maybeWriteString(modeJson, "name", mode.name);

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

bool AvailableSettings::isAvailable(const std::string &name, SettingType type) const{
	return vectorContains(available[type], name);
}

std::vector<std::string> AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

const std::vector<Device>& AvailableSettings::getDevices(SettingType type) const{
	return devices[type];
}

