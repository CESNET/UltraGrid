#include "combobox_ui.hpp"

ComboBoxUi::ComboBoxUi(QComboBox *box,
        Settings *settings,
        const std::string &opt,
        std::function<std::vector<SettingItem>()> itemBuilder) :
    WidgetUi(settings, opt),
    box(box),
    itemBuilder(itemBuilder),
    ignoreCallback(false)
{
    refresh();
    registerCallback();
    connectSignals();
}

void ComboBoxUi::connectSignals(){
	connect(box, QOverload<int>::of(&QComboBox::activated),
			this, &ComboBoxUi::itemSelected);
}

void ComboBoxUi::refresh(){
    items = itemBuilder();

    for(const auto &item : items){
        for(const auto &opt : item.opts){
            registerCallback(opt.opt);
        }
    }

    updateUiItems();
    selectOption();
}

static bool conditionsSatisfied(
        const std::vector<std::vector<ConditionItem>> &conds,
        Settings *settings)
{
    for(const auto &condClause : conds){
        bool orRes = false;
        for(const auto &condItem : condClause){
            bool val = condItem.value.val == settings->getOption(condItem.value.opt).getValue();

            //negate result if negation is true
            val = val != condItem.negation;

            if(val){
                orRes = true;
                break;
            }
        }
        if(!orRes){
            return false;
        }
    }

    return true;
}

void ComboBoxUi::selectOption(){
    int i;
    bool found = false;
    for(i = 0; i < box->count(); i++){
        const SettingItem &item = box->itemData(i).value<SettingItem>();

        found = true;
        for(const auto &itemOpt : item.opts){
            if(itemOpt.val != settings->getOption(itemOpt.opt).getValue()){
                found = false;
                break;
            }
        }
        
        if(found)
            break;
    }

    if(found){
        box->setCurrentIndex(i);
    } else {
        itemSelected(0);
    }
}

void ComboBoxUi::optChangeCallback(Option &opt, bool suboption){
    if(suboption || ignoreCallback)
        return;

    if(opt.getName() != this->opt)
        updateUiItems();

    selectOption();
}

void ComboBoxUi::updateUiState(){
    selectOption();
}

void ComboBoxUi::updateUiItems(){
    box->clear();

    for(const auto &i : items){
        if(conditionsSatisfied(i.conditions, settings)){
            box->addItem(QString::fromStdString(i.name),
                    QVariant::fromValue(i));
        }
    }
}

void ComboBoxUi::itemSelected(int index){
	const SettingItem &item = box->itemData(index).value<SettingItem>();

    ignoreCallback = true;
	for(const auto &option : item.opts){
		settings->getOption(option.opt).setValue(option.val);
	}
    ignoreCallback = false;

    emit changed();
}

