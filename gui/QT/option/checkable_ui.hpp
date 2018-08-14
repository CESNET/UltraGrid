#ifndef CHECKABLE_UI_HPP
#define CHECKABLE_UI_HPP

#include <string>

#include "widget_ui.hpp"
#include "settings.hpp"

class CheckableUi : public WidgetUi{
    Q_OBJECT

public:
    CheckableUi(Settings *settings, const std::string &opt);

    virtual ~CheckableUi() {  }

protected:
    virtual void connectSignals() override = 0;
    bool getOptValue();
    void updateUiState() override = 0;
    virtual void updateUiState(bool checked) = 0;

	void optChangeCallback(Option &opt, bool suboption) override;

protected slots:
    void boxClicked(bool checked);
};

#endif
