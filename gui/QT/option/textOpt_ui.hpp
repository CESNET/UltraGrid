#ifndef TEXTOPT_UI_HPP
#define TEXTOPT_UI_HPP

#include <string>
#include <QString>

#include "widget_ui.hpp"
#include "settings.hpp"

class TextOptUi : public WidgetUi{
    Q_OBJECT

public:
    TextOptUi(Settings *settings, const std::string &opt);

    virtual ~TextOptUi() {  }

protected:
    virtual void connectSignals() override = 0;
    std::string getOptValue();
    void updateUiState() override = 0;
    virtual void updateUiState(const std::string &str) = 0;

	void optChangeCallback(Option &opt, bool suboption) override;

protected slots:
    void textEdited(const QString &str);
};

#endif
