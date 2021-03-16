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

void WidgetUi::registerCustomCallback(const std::string &option,
		Option::Callback callback,
		std::unique_ptr<ExtraCallbackData>&& extraDataPtr)
{
	if(option == "")
		return;

	std::pair<std::string, Option::Callback> elToInsert(option, callback);

	auto it = std::find(registeredCallbacks.begin(),
			registeredCallbacks.end(), elToInsert);
	if(it != registeredCallbacks.end()){
		return;
	}


	settings->getOption(option).addOnChangeCallback(callback);
	registeredCallbacks.push_back(elToInsert);

	if(extraDataPtr)
		extraData.emplace_back(std::move(extraDataPtr));

}

void WidgetUi::registerCallback(const std::string &option){
	if(option == "")
		return;

	Option::Callback callback(&WidgetUi::optChangeCallbackStatic, this);
	registerCustomCallback(option, callback);
}

void WidgetUi::registerCallback(){
	registerCallback(opt);
}
