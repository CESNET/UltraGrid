#include "radioButton_ui.hpp"

RadioButtonUi::RadioButtonUi(QAbstractButton *btn,
        const std::string &selectedVal,
        Settings *settings,
        const std::string &opt) : 
    WidgetUi(settings, opt),
    btn(btn),
    selectedVal(selectedVal)
{
    updateUiState();
    connectSignals();
    registerCallback();
}

void RadioButtonUi::connectSignals(){
	connect(btn, &QAbstractButton::clicked, this, &RadioButtonUi::btnClicked);
}

void RadioButtonUi::updateUiState(bool checked){
    btn->setChecked(checked);
}

void RadioButtonUi::updateUiState(){
    updateUiState(settings->getOption(opt).getValue() == selectedVal);
}

void RadioButtonUi::optChangeCallback(Option &changedOpt, bool suboption){
    if(changedOpt.getName() == opt){
        updateUiState(settings->getOption(opt).getValue() == selectedVal);
    }
}

void RadioButtonUi::btnClicked(){
	settings->getOption(opt).setValue(selectedVal);
	emit changed();
}
