#ifndef SETTINGS_UI_HPP
#define SETTINGS_UI_HPP

#include <QObject>
#include <QString>
#include <memory>

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

	static void refreshAllCallback(Option&, bool, void *);

	void addCallbacks();

	void addControl(WidgetUi *widget);

private slots:

	void test();

	void buildSettingsCodecList();
	void settingsCodecSelected(QListWidgetItem *curr, QListWidgetItem *prev);
	void buildCodecOptControls(const std::string& mod, const std::string& codec);
	

signals:
	void changed();
};


#endif //SETTINGS_UI
