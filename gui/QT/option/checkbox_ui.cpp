#include "checkbox_ui.hpp"

CheckboxUi::CheckboxUi(QAbstractButton *box, Settings *settings, const std::string &opt) :
    CheckableUi(settings, opt),
    checkbox(box)
{
    updateUiState();
    connectSignals();
}

void CheckboxUi::connectSignals(){
	connect(checkbox, &QAbstractButton::clicked, this, &CheckboxUi::boxClicked);
}

void CheckboxUi::updateUiState(bool checked){
    checkbox->setChecked(checked);
}

void CheckboxUi::updateUiState(){
    updateUiState(getOptValue());
}

