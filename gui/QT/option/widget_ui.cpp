#include <functional>
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

void WidgetUi::registerCallback(const std::string &option){
    if(option == "" || registeredCallbacks.find(option) != registeredCallbacks.end())
        return;

    settings->getOption(option).addOnChangeCallback(
            std::bind(
                &WidgetUi::optChangeCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2));

    registeredCallbacks.insert(option);
}

void WidgetUi::registerCallback(){
    registerCallback(opt);
}
