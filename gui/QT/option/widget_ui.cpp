#include "widget_ui.hpp"

WidgetUi::WidgetUi(Settings *settings, const std::string &opt) :
    settings(settings),
    opt(opt)
{

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

    settings->getOption(option).addOnChangeCallback(
			Option::Callback(&WidgetUi::optChangeCallbackStatic, this));

    registeredCallbacks.insert(option);
}

void WidgetUi::registerCallback(){
    registerCallback(opt);
}
