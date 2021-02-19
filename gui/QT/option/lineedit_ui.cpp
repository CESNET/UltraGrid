#include "lineedit_ui.hpp"

LineEditUi::LineEditUi(QLineEdit *line, Settings *settings, const std::string &opt) : 
    TextOptUi(settings, opt),
    line(line)
{
    updateUiState();
    connectSignals();
}

void LineEditUi::connectSignals(){
	connect(line, &QLineEdit::textEdited, this, &LineEditUi::textEdited);
}

void LineEditUi::updateUiState(const std::string &text){
    line->setText(QString::fromStdString(text));
}

void LineEditUi::updateUiState(){
    updateUiState(getOptValue());
}

void LineEditUi::setEnabled(bool enabled){
	line->setEnabled(enabled);
}

void LineEditUi::setToolTip(const QString& toolTip){
	line->setToolTip(toolTip);
}


