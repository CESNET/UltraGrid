#include "checkable_ui.hpp"

CheckableUi::CheckableUi(Settings *settings, const std::string &opt) :
    WidgetUi(settings, opt)
{
    registerCallback();
}

void CheckableUi::boxClicked(bool checked){
	settings->getOption(opt).setValue(checked ? "t" : "f");
	//settings->getOption(opt).setEnabled(checked);
	emit changed();
}

void CheckableUi::optChangeCallback(Option &changedOpt, bool /*suboption*/){
    if(changedOpt.getName() == opt){
        updateUiState(changedOpt.isEnabled());
    }
}

bool CheckableUi::getOptValue(){
    return settings->getOption(opt).isEnabled();
}
