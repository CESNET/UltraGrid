#include <iostream>
#include <algorithm>
#include "settings.hpp"

Option::Callback::Callback(fcn_type func_ptr, void *opaque) :
	func_ptr(func_ptr),
	opaque(opaque)
{

}

void Option::Callback::operator()(Option& opt, bool suboption) const{
	func_ptr(opt, suboption, opaque);
}

bool Option::Callback::operator==(const Callback& o) const{
	return func_ptr == o.func_ptr && opaque == o.opaque;
}

std::string Option::getName() const{
	return name;
}

std::string Option::getValue() const{
	return value;
}

std::string Option::getParam() const{
	return param;
}

std::string Option::getSubVals() const{
	std::string out;

	for(const auto &sub : suboptions){
		if(sub.first.empty() || sub.first == value){
			out += sub.second->getLaunchOption();
		}
	}

	return out;
}

std::string Option::getLaunchOption() const{
	if(type == OptType::BoolOpt){
		return enabled ? param : "";
	}

	if(!enabled || value.empty() || type == OptType::SilentOpt)
		return "";

	std::string out = param + value;
	out += getSubVals();

	return out;
}

void Option::changed(){
	for(const auto &callback : onChangeCallbacks){
		callback(*this, false);
	}
}

void Option::suboptionChanged(Option &opt, bool, void *opaque){
	Option *obj = static_cast<Option *>(opaque);
	for(const auto &callback : obj->onChangeCallbacks){
		callback(opt, true);
	}
}

void Option::setValue(const std::string &val, bool suppressCallback){
	value = val;

	if(type == OptType::BoolOpt){
		enabled = val == "t";
	} else {
		enabled = !val.empty();
	}

	if(!suppressCallback)
		changed();
}

void Option::setDefaultValue(const std::string &val){
	defaultValue = val;
}

void Option::setEnabled(bool enable, bool suppressCallback){
	enabled = enable;
	if(!suppressCallback)
		changed();
}

void Option::addSuboption(Option *sub, const std::string &limit){
	sub->addOnChangeCallback(Callback(&Option::suboptionChanged, this));
	suboptions.push_back(std::make_pair(limit, sub));
}

void Option::addOnChangeCallback(Callback callback){
	onChangeCallbacks.push_back(callback);
}

void Option::removeOnChangeCallback(const Callback& callback){
	auto it = std::find(onChangeCallbacks.begin(), onChangeCallbacks.end(), callback);

	if(it == onChangeCallbacks.end()){
		std::cout << "Attempted to remove non-existent callback "
			<< name
			<< std::endl;
		return;
	}

#ifdef DEBUG
	std::cout << "Removing callback "
		<< name
		<< std::endl;
#endif
	onChangeCallbacks.erase(it);
}

Settings *Option::getSettings(){
	return settings;
}

#ifdef DEBUG
static void test_callback(Option &opt, bool, void *){
	std::cout << "Callback: " << opt.getName()
		<< ": " << opt.getValue()
		<< " (" << opt.isEnabled() << ")" << std::endl;
}
#endif

static void fec_builder_callback(Option &opt, bool subopt, void *){
	if(!subopt)
		return;

	Settings *settings = opt.getSettings();
	Option &fecOpt = settings->getOption("network.fec");
	bool enabled = settings->getOption("network.fec.enabled").isEnabled();

	std::string type = settings->getOption("network.fec.type").getValue();

	if(type.empty() || !enabled){
		fecOpt.setValue("");
	} else if(type == "mult"){
		fecOpt.setValue("mult:" + settings->getOption("network.fec.mult.factor").getValue());
	} else if(type == "rs"){
		fecOpt.setValue("rs:"
				+ settings->getOption("network.fec.rs.k").getValue()
				+ ":" + settings->getOption("network.fec.rs.n").getValue());
	} else if(type == "ldgm"){
		fecOpt.setValue("ldgm:"
				+ settings->getOption("network.fec.ldgm.k").getValue()
				+ ":" + settings->getOption("network.fec.ldgm.m").getValue()
				+ ":" + settings->getOption("network.fec.ldgm.c").getValue()
				+ settings->getOption("network.fec.ldgm.device").getParam()
				+ settings->getOption("network.fec.ldgm.device").getValue());
	}


}

static void fec_auto_callback(Option &opt, bool /*subopt*/, void *){
	Settings *settings = opt.getSettings();

	if(!settings->getOption("network.fec.auto").isEnabled())
		return;

	Option &fecOpt = settings->getOption("network.fec.type");
	const std::string &video_compress = settings->getOption("video.compress").getValue();

	if(video_compress == ""){
		fecOpt.setValue("");
	} else if(video_compress == "jpeg"){
		fecOpt.setValue("ldgm");
	} else if(video_compress == "libavcodec"
			&& settings->getOption("video.compress.libavcodec.codec").getValue() == "MJPEG"){
		fecOpt.setValue("ldgm");
	} else {
		fecOpt.setValue("rs");
	}
}

const static struct{
	const char *name;
	Option::OptType type;
	const char *param;
	const char *defaultVal;
	bool enabled;
	const char *parent;
	const char *limit;
} optionList[] = {
	{"video.source", Option::StringOpt, " -t ", "", false, "", ""},
	{"testcard.width", Option::StringOpt, ":", "", false, "video.source", "testcard"},
	{"testcard.height", Option::StringOpt, ":", "", false, "video.source", "testcard"},
	{"testcard.fps", Option::StringOpt, ":", "", false, "video.source", "testcard"},
	{"testcard.format", Option::StringOpt, ":", "", false, "video.source", "testcard"},
	{"screen.fps", Option::StringOpt, ":fps=", "", false, "video.source", "screen"},
	{"v4l2.device", Option::StringOpt, ":device=", "", false, "video.source", "v4l2"},
	{"v4l2.codec", Option::StringOpt, ":codec=", "", false, "video.source", "v4l2"},
	{"v4l2.size", Option::StringOpt, ":size=", "", false, "video.source", "v4l2"},
	{"v4l2.tpf", Option::StringOpt, ":tpf=", "", false, "video.source", "v4l2"},
	{"v4l2.force_rgb", Option::BoolOpt, ":RGB", "t", true, "video.source", "v4l2"},
	{"dshow.device", Option::StringOpt, ":device=", "", false, "video.source", "dshow"},
	{"dshow.mode", Option::StringOpt, ":mode=", "", false, "video.source", "dshow"},
	{"dshow.force_rgb", Option::BoolOpt, ":RGB", "t", true, "video.source", "dshow"},
	{"avfoundation.device", Option::StringOpt, ":device=", "", false, "video.source", "avfoundation"},
	{"avfoundation.mode", Option::StringOpt, ":mode=", "", false, "video.source", "avfoundation"},
	{"avfoundation.fps", Option::StringOpt, ":fps=", "", false, "video.source", "avfoundation"},
	{"decklink.modeOpt", Option::StringOpt, ":", "", false, "video.source", "decklink"},
	{"decklink.device", Option::StringOpt, ":device=", "", false, "video.source", "decklink"},
	{"video.display", Option::StringOpt, " -d ", "", false, "", ""},
	{"gl.novsync", Option::BoolOpt, ":novsync", "f", false, "video.display", "gl"},
	{"gl.deinterlace", Option::BoolOpt, ":d", "f", false, "video.display", "gl"},
	{"gl.fullscreen", Option::BoolOpt, ":fs", "f", false, "video.display", "gl"},
	{"gl.nodecorate", Option::BoolOpt, ":nodecorate", "f", false, "video.display", "gl"},
	{"gl.cursor", Option::BoolOpt, ":cursor", "f", false, "video.display", "gl"},
	{"sdl.deinterlace", Option::BoolOpt, ":d", "f", false, "video.display", "sdl"},
	{"sdl.fullscreen", Option::BoolOpt, ":fs", "f", false, "video.display", "sdl"},
	{"video.compress", Option::StringOpt, " -c ", "", false, "", ""},
	{"libavcodec.codec", Option::StringOpt, ":codec=", "", false, "video.compress", "libavcodec"},
	{"H.264.bitrate", Option::StringOpt, ":bitrate=", "", false, "video.compress.libavcodec.codec", "H.264"},
	{"H.265.bitrate", Option::StringOpt, ":bitrate=", "", false, "video.compress.libavcodec.codec", "H.265"},
	{"MJPEG.bitrate", Option::StringOpt, ":bitrate=", "", false, "video.compress.libavcodec.codec", "MJPEG"},
	{"VP8.bitrate", Option::StringOpt, ":bitrate=", "", false, "video.compress.libavcodec.codec", "VP8"},
	{"jpeg.quality", Option::StringOpt, ":", "", false, "video.compress", "jpeg"},
	{"audio.source", Option::StringOpt, " -s ", "", false, "", ""},
	{"audio.source.channels", Option::StringOpt, " --audio-capture-format channels=", "", false, "", ""},
	{"audio.playback", Option::StringOpt, " -r ", "", false, "", ""},
	{"audio.compress", Option::StringOpt, " --audio-codec ", "", false, "", ""},
	{"bitrate", Option::StringOpt, ":bitrate=", "", false, "audio.compress", ""},
	{"network.destination", Option::StringOpt, " ", "", false, "", ""},
	{"network.port", Option::StringOpt, " -P ", "5004", false, "", ""},
	{"network.control_port", Option::StringOpt, " --control-port ", "8888", true, "", ""},
	{"network.fec", Option::StringOpt, " -f ", "", false, "", ""},
	{"type", Option::SilentOpt, "", "", false, "network.fec", ""},
	{"enabled", Option::BoolOpt, "", "f", false, "network.fec", ""},
	{"mult.factor", Option::SilentOpt, ":", "2", false, "network.fec", ""},
	{"ldgm.k", Option::SilentOpt, ":", "256", false, "network.fec", ""},
	{"ldgm.m", Option::SilentOpt, ":", "192", false, "network.fec", ""},
	{"ldgm.c", Option::SilentOpt, ":", "5", false, "network.fec", ""},
	{"ldgm.device", Option::SilentOpt, " --param ldgm-device=", "CPU", false, "network.fec", ""},
	{"rs.k", Option::SilentOpt, ":", "200", false, "network.fec", ""},
	{"rs.n", Option::SilentOpt, "", "220", false, "network.fec", ""},
	{"network.fec.auto", Option::BoolOpt, "", "t", true, "", ""},
	{"decode.hwaccel", Option::BoolOpt, " --param use-hw-accel", "f", false, "", ""},
	{"advanced", Option::BoolOpt, "", "f", false, "", ""},
	{"preview", Option::BoolOpt, "", "t", true, "", ""},
	{"vuMeter", Option::BoolOpt, "", "t", true, "", ""},
	{"errors_fatal", Option::BoolOpt, " --param errors-fatal", "t", true, "", ""},
	{"aja.device", Option::StringOpt, ":", "", false, "video.source", "aja"},
	{"bluefish444.device", Option::StringOpt, ":", "", false, "video.source", "bluefish444"},
	{"deltacast.device", Option::StringOpt, ":", "", false, "video.source", "deltacast"},
	{"deltacast-dv.device", Option::StringOpt, ":", "", false, "video.source", "deltacast-dv"},
	{"dvs.device", Option::StringOpt, ":", "", false, "video.source", "dvs"},
	{"ndi.device", Option::StringOpt, ":", "", false, "video.source", "ndi"},
	{"spout.device", Option::StringOpt, ":", "", false, "video.source", "spout"},
	{"syphon.device", Option::StringOpt, ":", "", false, "video.source", "syphon"},
	{"ximea.device", Option::StringOpt, ":", "", false, "video.source", "ximea"},
	{"encryption", Option::StringOpt, " --encryption ", "", false, "", ""},
};

const struct {
	const char *name;
	Option::Callback::fcn_type callback;
} optionCallbacks[] = {
	{"network.fec", fec_builder_callback},
	{"video.compress", fec_auto_callback},
	{"network.fec.auto", fec_auto_callback},
};

Settings::Settings() : dummy(this){
	for(const auto &i : optionList){
		auto &opt = addOption(i.name,
				i.type,
				i.param,
				i.defaultVal,
				i.enabled,
				i.parent,
				i.limit);
#ifdef DEBUG
		if(!i.parent[0])
			opt.addOnChangeCallback(test_callback);
#endif
		(void) opt; //suppress unused warning
	}

	for(const auto &i : optionCallbacks){
		getOption(i.name).addOnChangeCallback(Option::Callback(i.callback, nullptr));
	}

	std::cout << getLaunchParams() << std::endl;
}

std::string Settings::getLaunchParams() const{
	std::string out;

	if(getOption("preview").isEnabled()){
		out += "--capture-filter preview";
	}

	out += getOption("video.source").getLaunchOption();
	out += getOption("video.compress").getLaunchOption();
	const auto &dispOpt = getOption("video.display");
	if(dispOpt.isEnabled()){
		if(getOption("preview").isEnabled()){
			out += dispOpt.getParam();
			out += "multiplier:" + dispOpt.getValue() + dispOpt.getSubVals();
			out += "#preview";
		} else {
			out += getOption("video.display").getLaunchOption();
		}
	} else {
		if(getOption("preview").isEnabled()){
			out += dispOpt.getParam();
			out += "preview";
		}
	}

	out += getOption("audio.source").getLaunchOption();
	out += getOption("audio.source.channels").getLaunchOption();
	out += getOption("audio.compress").getLaunchOption();
	std::string audioPlay = getOption("audio.playback").getLaunchOption();
	if(audioPlay.empty() && getOption("preview").isEnabled()){
		audioPlay = " -r dummy";
	}
	out += audioPlay;
	out += getOption("network.fec").getLaunchOption();
	out += getOption("encryption").getLaunchOption();
	out += getOption("network.port").getLaunchOption();
	out += getOption("network.control_port").getLaunchOption();
	out += getOption("network.destination").getLaunchOption();
	out += getOption("decode.hwaccel").getLaunchOption();
	out += getOption("errors_fatal").getLaunchOption();
	return out;
}

std::string Settings::getPreviewParams() const{
	std::string out;

	out += " --capture-filter preview,every:0";
	out += getOption("video.source").getLaunchOption();
	out += " -d preview";
	out += getOption("audio.source").getLaunchOption();
	out += getOption("audio.source.channels").getLaunchOption();
	out += " -r dummy";
	out += getOption("network.control_port").getLaunchOption();
	out += getOption("encryption").getLaunchOption();

	return out;
}

Option& Settings::getOption(const std::string &opt){
	auto search = options.find(opt);

	if(search == options.end()){
		std::unique_ptr<Option> newOption(new Option(this, opt));
		auto p = newOption.get();
		options[opt] = std::move(newOption);
		return *p;
	}

	return *search->second;
}

const Option& Settings::getOption(const std::string &opt) const{
	auto search = options.find(opt);

	if(search == options.end())
		return dummy;

	return *search->second;
}

const std::map<std::string, std::unique_ptr<Option>>& Settings::getOptionMap() const{
	return options;
}

Option& Settings::addOption(std::string name,
		Option::OptType type,
		const std::string &param,
		const std::string &value,
		bool enabled,
		const std::string &parent,
		const std::string &limit)
{
	if(!parent.empty())
		name = parent + "." + name;

	auto search = options.find(name);

	Option *opt;

	if(search == options.end()){
		std::unique_ptr<Option> newOption(new Option(this, name));
		opt = newOption.get();
		options[name] = std::move(newOption);
	} else {
		opt = search->second.get();
	}

	opt->setParam(param);
	opt->setType(type);
	opt->setValue(value);
	opt->setDefaultValue(value);
	opt->setEnabled(enabled);

	if(!parent.empty()){
		getOption(parent).addSuboption(opt, limit);
	}

	return *opt;
}

bool Settings::isAdvancedMode(){
	return getOption("advanced").getValue() == "t";
}

void Settings::changedAll(){
	for(auto &i : options){
		i.second->changed();
	}
}

/* vim: set noexpandtab: */
