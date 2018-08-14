#ifndef GROUPBOX_UI_HPP
#define GROUPBOX_UI_HPP

#include <QGroupBox>
#include <string>

#include "checkable_ui.hpp"
#include "settings.hpp"

class GroupBoxUi : public CheckableUi{
Q_OBJECT
public:
    GroupBoxUi(QGroupBox *groupBox, Settings *settings, const std::string &opt);

protected:
    QGroupBox *groupBox;

    void connectSignals() override;
    void updateUiState(bool checked) override;
    void updateUiState() override;

};

#endif
