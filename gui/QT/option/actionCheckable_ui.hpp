#ifndef ACTIONCHECKABLE_UI_HPP
#define ACTIONCHECKABLE_UI_HPP

#include <QAction>
#include <string>

#include "checkable_ui.hpp"
#include "settings.hpp"

class ActionCheckableUi : public CheckableUi{
    Q_OBJECT

public:
    ActionCheckableUi(QAction *action, Settings *settings, const std::string &opt);

protected:
    QAction *action;

    void connectSignals() override;
    void updateUiState(bool checked) override;
    void updateUiState() override;
};

#endif
