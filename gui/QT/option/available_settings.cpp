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
#include "utils/string_view_utils.hpp"

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

static QString getProcessOutput(const std::string& executable, const QStringList& args){
	QProcess process;
	process.start(executable.c_str(), args);
	process.waitForFinished();

	QString output = QString(process.readAllStandardOutput());

	return output;
}

struct {
	SettingType type;
	std::string_view capStr;
} modulesQueryInfo[] = {
	{VIDEO_SRC, "[cap][capture] "},
	{VIDEO_DISPLAY, "[cap][display] "}, 
	{VIDEO_COMPRESS, "[cap][compress] "},
	{VIDEO_CAPTURE_FILTER, "[cap][capture_filter] "},
	{AUDIO_SRC, "[cap][audio_cap] "},
	{AUDIO_PLAYBACK, "[cap][audio_play] "},
	{AUDIO_COMPRESS, "[cap][audio_compress] "},
};

void AvailableSettings::queryProcessLine(std::string_view line){
	for(const auto& i : modulesQueryInfo){
		if(sv_is_prefix(line, i.capStr)){
			auto mod = line.substr(i.capStr.size());
			auto cutIdx = mod.find_first_of("\n \r");
			if(cutIdx != mod.npos)
				mod.remove_suffix(mod.size() - cutIdx);
			available[i.type].emplace_back(mod);
			return;
		}
	}

	constexpr std::string_view devStr = "[capability][device]";
	constexpr std::string_view codecStr = "[capability][video_compress]";
	constexpr std::string_view endMarkerStr = "[capability][end]";

	if(sv_is_prefix(line, devStr)){
		line.remove_prefix(devStr.size());
		queryDevice(line);
	} else if(sv_is_prefix(line, codecStr)){
		line.remove_prefix(codecStr.size());
		queryVideoCompress(line);
	} else if(sv_is_prefix(line, endMarkerStr)){
		endMarkerFound = true;
	}
}

void AvailableSettings::parseStateVersion(std::string_view line){
	int ver = 0;
	std::string_view verStr = "[capability][start] version ";

	if(sv_is_prefix(line, verStr)){
		line.remove_prefix(verStr.size());
		parse_num(line, ver);

		const int expectedVersion = 4;
		if(ver != expectedVersion){
			QMessageBox msgBox;
			QString msg = "Capabilities start marker with expected version not found"
				" in ug output.";
			if(ver != 0)
				msg += " Version found: " + QString::number(ver);
			msgBox.setText(msg);
			msgBox.setIcon(QMessageBox::Critical);
			msgBox.exec();
			parseFunc = nullptr;
		} else {
			parseFunc = &AvailableSettings::queryProcessLine;
		}
	}
}

void AvailableSettings::queryBegin(){
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

	endMarkerFound = false;
	parseFunc = &AvailableSettings::parseStateVersion;
}

void AvailableSettings::queryEnd(){
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

void AvailableSettings::queryLine(std::string_view line){
	if(!parseFunc || (!sv_is_prefix(line, "[cap]") && !sv_is_prefix(line, "[capability]")))
		return;

	(this->*parseFunc)(line);
}

void AvailableSettings::queryFromString(const QString &string){
	QStringList lines = string.split(QRegularExpression("\n|\r\n|\r"));

	queryBegin();

	for(int i = 0; i < lines.size(); i++){
		auto stdStr = lines[i].toStdString();
		queryLine(stdStr);
	}

	queryEnd();
}

static void maybeWriteString (const QJsonObject& obj,
		const char *key,
		std::string& result)
{
		if(obj.contains(key) && obj[key].isString()){
			result = obj[key].toString().toStdString();
		}
}

void AvailableSettings::queryAll(const std::string &executable){
	QString output = getProcessOutput(executable, QStringList() << "--capabilities");
	queryFromString(output);
}

void AvailableSettings::queryVideoCompress(std::string_view line){
	QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromRawData(line.data(), line.size()));
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

void AvailableSettings::queryDevice(std::string_view line){
	static std::map<std::string, SettingType> settingTypeMap = {
		{"video_cap", VIDEO_SRC},
		{"video_disp", VIDEO_DISPLAY},
		{"audio_play", AUDIO_PLAYBACK},
		{"audio_cap", AUDIO_SRC},
	};

	QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromRawData(line.data(), line.size()));
	if(!doc.isObject())
		return;

	Device dev;
	QJsonObject obj = doc.object();
	maybeWriteString(obj, "name", dev.name);
	maybeWriteString(obj, "module", dev.type);
	maybeWriteString(obj, "device", dev.deviceOpt);

	if(obj.contains("extra") && obj["extra"].isObject()){
		const auto& extraObj = obj["extra"].toObject();
		for(auto it = extraObj.constBegin(); it != extraObj.constEnd(); it++){
			if(it.value().isString()){
				dev.extra[it.key().toStdString()] = it.value().toString().toStdString();
			}
		}
	}

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

