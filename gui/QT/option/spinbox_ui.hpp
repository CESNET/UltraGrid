#ifndef SPINBOX_UI_HPP
#define SPINBOX_UI_HPP

#include <QSpinBox>
#include "textOpt_ui.hpp"

class SpinBoxUi : public TextOptUi{
Q_OBJECT

public:
    SpinBoxUi(QSpinBox *line, Settings *settings, const std::string &opt);

private:
    QSpinBox *spinbox;

    void connectSignals() override;
    void updateUiState() override;
    void updateUiState(const std::string &text);
};

#endif
