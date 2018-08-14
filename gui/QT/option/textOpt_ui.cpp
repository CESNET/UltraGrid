#include "textOpt_ui.hpp"

TextOptUi::TextOptUi(Settings *settings, const std::string &opt) :
    WidgetUi(settings, opt)
{
    registerCallback();
}

void TextOptUi::textEdited(const QString &str){
	settings->getOption(opt).setValue(str.toStdString());
	emit changed();
}

void TextOptUi::optChangeCallback(Option &changedOpt, bool suboption){
    if(changedOpt.getName() == opt){
        updateUiState(changedOpt.getValue());
    }
}

std::string TextOptUi::getOptValue(){
    return settings->getOption(opt).getValue();
}
