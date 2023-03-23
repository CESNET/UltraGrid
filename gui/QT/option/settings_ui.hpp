#ifndef SETTINGS_UI_HPP
#define SETTINGS_UI_HPP

#include <QObject>
#include <QString>
#include <QFormLayout>
#include <QWidget>
#include <memory>
#include <map>

#include "available_settings.hpp"
#include "ui_ultragrid_window.h"
#include "ui_settings.h"
#include "settings.hpp"
#include "checkbox_ui.hpp"
#include "actionCheckable_ui.hpp"
#include "combobox_ui.hpp"
#include "lineedit_ui.hpp"
#include "spinbox_ui.hpp"
#include "groupBox_ui.hpp"
#include "radioButton_ui.hpp"
#include "audio_opts.hpp"
#include "video_opts.hpp"

struct ControlForm{
	ControlForm() : uiContainer(new QWidget()) {
		formLayout = new QFormLayout(uiContainer.get());
		uiContainer->setLayout(formLayout);
	}
	std::vector<std::unique_ptr<WidgetUi>> uiControls;
	std::unique_ptr<QWidget> uiContainer;
	QFormLayout *formLayout = nullptr; //owned by uiContainer
};

struct DeviceOptTab{
	std::unique_ptr<QWidget> page;
	QListWidget *listWidget = nullptr;
	QScrollArea *scrollArea = nullptr;

	std::vector<std::unique_ptr<WidgetUi>> uiControls;
	std::string keyPrefix;
};

class SettingsUi : public QObject{
	Q_OBJECT

public:
	void init(Settings *settings, AvailableSettings *availableSettings);
	void initMainWin(Ui::UltragridWindow *ui);
	void initSettingsWin(Ui::Settings *ui);

	void refreshAll();

private:
	Ui::UltragridWindow *mainWin = nullptr;
	Ui::Settings *settingsWin = nullptr;
	Settings *settings = nullptr;
	AvailableSettings *availableSettings = nullptr;

	std::vector<std::unique_ptr<WidgetUi>> uiControls;
	std::vector<std::unique_ptr<WidgetUi>> codecControls;
	std::vector<std::unique_ptr<WidgetUi>> displayControls;

	std::map<SettingType, DeviceOptTab> deviceTabs;

	static void refreshAllCallback(Option&, bool, void *);

	void addCallbacks();

	void addControl(WidgetUi *widget);

	void addDeviceTabs();
	void buildSettingsDeviceList(QListWidget *list, SettingType type);
	void buildDeviceOptControls(const Device& dev, DeviceOptTab& tab);

	void fillFormOptions(ControlForm& form,
			std::string_view keyPrefix,
			const std::vector<CapabOpt>& opts);

private slots:

	void test();

	void buildSettingsCodecList();
	void settingsCodecSelected(QListWidgetItem *curr, QListWidgetItem *prev);
	void buildCodecOptControls(const Codec& codec);

	void settingsDeviceSelected(QListWidgetItem *curr, QListWidgetItem *prev);

signals:
	void changed();
};


#endif //SETTINGS_UI
