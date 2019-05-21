#include "spinbox_ui.hpp"
#include "overload.hpp"

SpinBoxUi::SpinBoxUi(QSpinBox *spinbox, Settings *settings, const std::string &opt) : 
    TextOptUi(settings, opt),
    spinbox(spinbox)
{
    updateUiState();
    connectSignals();
}

void SpinBoxUi::connectSignals(){
	connect(spinbox, Overload<const QString &>::of(&QSpinBox::valueChanged),
            this, &SpinBoxUi::textEdited);
}

void SpinBoxUi::updateUiState(const std::string &text){
    if(!text.empty())
        spinbox->setValue(std::stoi(text));
    else
        spinbox->clear();
}

void SpinBoxUi::updateUiState(){
    updateUiState(getOptValue());
}
