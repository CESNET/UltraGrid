#ifndef COMBOBOX_UI_HPP
#define COMBOBOX_UI_HPP

#include <QComboBox>
#include <functional>
#include "widget_ui.hpp"

struct SettingValue{
	std::string opt;
	std::string val;
};

struct ConditionItem{
    SettingValue value;
    bool negation;
};

struct SettingItem{
	std::string name;
	std::vector<SettingValue> opts;

    //conditions in conjunctive normal form
	std::vector<std::vector<ConditionItem>> conditions;
};

Q_DECLARE_METATYPE(SettingItem);

class ComboBoxUi : public WidgetUi{
Q_OBJECT

public:
    ComboBoxUi(QComboBox *box,
            Settings *settings,
            const std::string &opt,
            std::function<std::vector<SettingItem>()> itemBuilder);

    void refresh() override;

private:
    QComboBox *box;
    std::function<std::vector<SettingItem>()> itemBuilder;
    bool ignoreCallback;

    std::vector<SettingItem> items;

    void connectSignals() override;
    void updateUiState() override;

    void updateUiItems();

	void optChangeCallback(Option &opt, bool suboption) override;

    void selectOption();

private slots:
    void itemSelected(int index);
};


#endif
