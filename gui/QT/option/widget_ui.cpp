#include "widget_ui.hpp"

WidgetUi::WidgetUi(Settings *settings, const std::string &opt) :
	settings(settings),
	opt(opt)
{
}

WidgetUi::~WidgetUi(){
	for(auto& i : registeredCallbacks){
		settings->getOption(i.first).removeOnChangeCallback(i.second);
	}
}

void WidgetUi::setOpt(const std::string &opt){
	this->opt = opt;
	updateUiState();
	registerCallback();
}

void WidgetUi::optChangeCallbackStatic(Option& opt, bool subopt, void *opaque){
	static_cast<WidgetUi *>(opaque)->optChangeCallback(opt, subopt);
}

void WidgetUi::registerCallback(const std::string &option){
	if(option == "" || registeredCallbacks.find(option) != registeredCallbacks.end())
		return;

	Option::Callback callback(&WidgetUi::optChangeCallbackStatic, this);
	settings->getOption(option).addOnChangeCallback(callback);

	registeredCallbacks.insert({option, callback});
}

void WidgetUi::registerCallback(){
	registerCallback(opt);
}
