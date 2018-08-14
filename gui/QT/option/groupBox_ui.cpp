#include "groupBox_ui.hpp"

GroupBoxUi::GroupBoxUi(QGroupBox *groupBox, Settings *settings, const std::string &opt) :
    CheckableUi(settings, opt),
    groupBox(groupBox)
{
    updateUiState();
    connectSignals();
}

void GroupBoxUi::connectSignals(){
	connect(groupBox, &QGroupBox::clicked, this, &GroupBoxUi::boxClicked);
}

void GroupBoxUi::updateUiState(bool checked){
    groupBox->setChecked(checked);
}

void GroupBoxUi::updateUiState(){
    updateUiState(getOptValue());
}

