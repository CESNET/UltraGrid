#ifndef CHECKBOX_UI_HPP
#define CHECKBOX_UI_HPP

#include <QAbstractButton>
#include <string>

#include "checkable_ui.hpp"
#include "settings.hpp"

class CheckboxUi : public CheckableUi{
Q_OBJECT

public:
	CheckboxUi(QAbstractButton *box, Settings *settings, const std::string &opt);

protected:
	QAbstractButton *checkbox;

	void connectSignals() override;
	void updateUiState(bool checked) override;
	void updateUiState() override;
};

#endif
