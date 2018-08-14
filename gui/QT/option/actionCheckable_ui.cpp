#include "actionCheckable_ui.hpp"

ActionCheckableUi::ActionCheckableUi(QAction *action, Settings *settings, const std::string &opt) :
    CheckableUi(settings, opt),
    action(action)
{
    updateUiState();
    connectSignals();
}

void ActionCheckableUi::connectSignals(){
	connect(action, &QAction::triggered, this, &ActionCheckableUi::boxClicked);
}

void ActionCheckableUi::updateUiState(bool checked){
    action->setChecked(checked);
}

void ActionCheckableUi::updateUiState(){
    updateUiState(getOptValue());
}

