#include <QProcess>
#include <QString>
#include <QRegularExpression>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <cstring>
#include "available_settings.hpp"

#include <iostream>

static bool vectorContains(const std::vector<std::string> &v, const std::string & s){
	for(unsigned i = 0; i < v.size(); i++){
		if(v[i] == s)
			return true;
	}
	return false;
}

static QStringList getProcessOutput(const std::string& executable, const std::string& command){
	QProcess process;
	process.start((executable + command).c_str());
	process.waitForFinished();

	QString output = QString(process.readAllStandardOutput());
#ifdef WIN32
	QString lineSeparator = "\r\n";
#else
	QString lineSeparator = "\n";
#endif

	QStringList lines = output.split(lineSeparator);

	return lines;
}

static std::vector<std::string> getAvailOpts(const std::string &opt, const std::string &executable){
	std::vector<std::string> out;

	std::string command;
	command += " ";
	command += opt;
	command += " help";

	QStringList lines = getProcessOutput(executable, command);

	foreach ( const QString &line, lines ) {
		if(line.size() > 0 && QChar(line[0]).isSpace()) {
			QString opt = line.trimmed();
			if(opt != "none"
					&& !opt.startsWith("--")
					&& !opt.contains("unavailable"))
				out.push_back(line.trimmed().toStdString());
		}
	}

	return out;
}

void AvailableSettings::queryAll(const std::string &executable){
	QStringList lines = getProcessOutput(executable, " --capabilities");

	queryCap(lines, VIDEO_SRC, "[cap][capture] ");
	queryCap(lines, VIDEO_DISPLAY, "[cap][display] ");
	queryCap(lines, VIDEO_COMPRESS, "[cap][compress] ");
	queryCap(lines, VIDEO_CAPTURE_FILTER, "[cap][capture_filter] ");
	queryCap(lines, AUDIO_SRC, "[cap][audio_cap] ");
	queryCap(lines, AUDIO_PLAYBACK, "[cap][audio_play] ");
	
	queryCapturers(lines);

	//TODO
	query(executable, AUDIO_COMPRESS);

#ifdef DEBUG
	for(const auto &i : capturers){
		std::cout << i.name << " : " << i.type
			<< " : " << i.deviceOpt << std::endl;

		for(const auto &j : i.modes){
			std::cout << "\t" << j.name << std::endl;
			for(const auto &k : j.opts){
				std::cout << "\t\t" << k.opt << " : " << k.val << std::endl;
			}
		}
	}
#endif
}

void AvailableSettings::queryCapturers(const QStringList &lines){
	const char * const capStr = "[capability][capture][v1]";
	size_t capStrLen = strlen(capStr);

	capturers.clear();

	foreach ( const QString &line, lines ) {
		if(line.startsWith(capStr)){
			QJsonDocument doc = QJsonDocument::fromJson(line.mid(capStrLen).toUtf8());
			if(!doc.isObject())
				return;

			Capturer cap;
			QJsonObject obj = doc.object();
			if(obj.contains("name") && obj["name"].isString()){
				cap.name = obj["name"].toString().toStdString();
			}
			if(obj.contains("type") && obj["type"].isString()){
				cap.type = obj["type"].toString().toStdString();
			}
			if(obj.contains("device") && obj["device"].isString()){
				cap.deviceOpt = obj["device"].toString().toStdString();
			}

			if(obj.contains("modes") && obj["modes"].isArray()){
				for(const QJsonValue &val : obj["modes"].toArray()){
					if(val.isObject()){
						QJsonObject modeJson = val.toObject();

						CaptureMode mode;

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
							cap.modes.push_back(std::move(mode));
						}
					}
				}
			}
			capturers.push_back(std::move(cap));
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

void AvailableSettings::query(const std::string &executable, SettingType type){
	std::string opt;

	switch(type){
		case VIDEO_SRC: opt = "-t"; break;
		case VIDEO_DISPLAY: opt = "-d"; break;
		case VIDEO_COMPRESS: opt = "-c"; break;
		case VIDEO_CAPTURE_FILTER: opt = "--capture-filter"; break;
		case AUDIO_SRC: opt = "-s"; break;
		case AUDIO_PLAYBACK: opt = "-r"; break;
		case AUDIO_COMPRESS: opt = "--audio-codec"; break;
		default: return;
	}

	available[type] = getAvailOpts(opt, executable);
}

bool AvailableSettings::isAvailable(const std::string &name, SettingType type) const{
	return vectorContains(available[type], name);
}

std::vector<std::string> AvailableSettings::getAvailableSettings(SettingType type) const{
	return available[type];
}

std::vector<Capturer> AvailableSettings::getCapturers() const{
	return capturers;
}

