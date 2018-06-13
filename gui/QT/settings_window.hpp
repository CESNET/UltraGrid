#ifndef SETTINGS_WINDOW
#define SETTINGS_WINDOW

#include <QString>
#include "ui_settings.h"

class SettingsWindow : public QDialog{
	Q_OBJECT

public:
	SettingsWindow(QWidget *parent = 0);
	QString getVideoPort() const;
	QString getAudioPort() const;
	bool isDefault() const;
	QString getPortArgs() const;

signals:
	void changed();


private:
	Ui::Settings ui;

};
#endif
