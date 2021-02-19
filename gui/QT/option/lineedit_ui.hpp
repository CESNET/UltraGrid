#ifndef LINEEDIT_UI_HPP
#define LINEEDIT_UI_HPP

#include <QLineEdit>
#include "textOpt_ui.hpp"

class LineEditUi : public TextOptUi{
Q_OBJECT

public:
    LineEditUi(QLineEdit *line, Settings *settings, const std::string &opt);

	void setEnabled(bool enabled);
	void setToolTip(const QString& toolTip);

private:
    QLineEdit *line;

    void connectSignals() override;
    void updateUiState() override;
    void updateUiState(const std::string &text);
};

#endif
