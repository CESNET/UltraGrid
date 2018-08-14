#ifndef RADIOBUTTON_UI_HPP
#define RADIOBUTTON_UI_HPP

#include <QAbstractButton>
#include <string>

#include "widget_ui.hpp"


class RadioButtonUi : public WidgetUi{
Q_OBJECT

public:
    RadioButtonUi(QAbstractButton *btn,
            const std::string &selectedVal,
            Settings *settings,
            const std::string &opt);

private:
    QAbstractButton *btn;
    std::string selectedVal;

    void connectSignals() override;
    void updateUiState() override;
    void updateUiState(bool checked);
	void optChangeCallback(Option &opt, bool suboption) override;

private slots:
    void btnClicked();

};

#endif
