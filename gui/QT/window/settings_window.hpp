#ifndef SETTINGS_WINDOW
#define SETTINGS_WINDOW

#include <QString>
#include "ui_settings.h"
#include "settings_ui.hpp"

class SettingsWindow : public QDialog{
	Q_OBJECT

public:
	SettingsWindow(QWidget *parent = 0);

	void init(SettingsUi *settingsUi, Settings *s);

signals:
	void changed();


private:
	Ui::Settings ui;
	Settings *settings;

private slots:
	void changeFecPage();
	void fecTab();

};
#endif
