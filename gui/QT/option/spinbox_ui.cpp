#include <QtGlobal>
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
#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
	connect(spinbox, Overload<const QString &>::of(&QSpinBox::valueChanged),
			this, &SpinBoxUi::textEdited);
#else
	connect(spinbox, &QSpinBox::textChanged,
            this, &SpinBoxUi::textEdited);
#endif
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
