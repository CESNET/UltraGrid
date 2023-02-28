#include <QString>
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

const char *settingTypeToStr(SettingType type){
#define X(type, str) case type: return str;
	switch(type){
	SETTING_TYPE_LIST
	case SETTING_TYPE_COUNT: return "";
	}
#undef X

	abort(); //Should be unreachable
}

SettingType strToSettingType(std::string_view sv){
#define X(type, str) if(sv == str) return type; else
	SETTING_TYPE_LIST{
		return SETTING_TYPE_UNKNOWN;
	}
#undef X

	abort(); //Should be unreachable
}

void AvailableSettings::queryProcessLine(std::string_view line){
	constexpr std::string_view module_cap_prefix = "[cap][";
	if(sv_is_prefix(line, module_cap_prefix)){
		line.remove_prefix(module_cap_prefix.size());
		if(line.empty())
			return;

		auto module_type = tokenize(line, ']');
		auto module_name = sv_trim_whitespace(tokenize(line, ']'));

		if(!module_type.empty() && !module_name.empty())
			available[strToSettingType(module_type)].emplace_back(module_name);

		return;
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

void AvailableSettings::queryBeginPass(){
	endMarkerFound = false;
	parseFunc = &AvailableSettings::parseStateVersion;
}

void AvailableSettings::queryBegin(){
	for(auto& i : available){
		i.clear();
	}
	for(auto& i : devices){
		i.clear();
	}
	videoCompressModules.clear();
	videoCompressCodecs.clear();
	videoCompressModules.emplace_back(CompressModule{"", {}, });
	videoCompressCodecs.emplace_back(Codec{"None", "", {Encoder{"default", ""}}, 0});

	queryBeginPass();
}

void AvailableSettings::queryEnd(){
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

static void maybeWriteString (const QJsonObject& obj,
		const char *key,
		std::string& result)
{
		if(obj.contains(key) && obj[key].isString()){
			result = obj[key].toString().toStdString();
		}
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

const std::vector<std::string>& AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

const std::vector<Device>& AvailableSettings::getDevices(SettingType type) const{
	return devices[type];
}

