#include <QIntValidator> 
#include "settings_window.hpp"

SettingsWindow::SettingsWindow(QWidget *parent): QDialog(parent){
	ui.setupUi(this);
	ui.videoPort->setValidator(new QIntValidator(0, 65535, this));
	ui.audioPort->setValidator(new QIntValidator(0, 65535, this));

	connect(ui.videoPort, SIGNAL(textEdited(const QString&)), this, SIGNAL(changed()));
	connect(ui.audioPort, SIGNAL(textEdited(const QString&)), this, SIGNAL(changed()));
}

QString SettingsWindow::getVideoPort() const {
	QString s = ui.videoPort->text();
	if(s == "")
		return "5004";

	return s;
}

QString SettingsWindow::getAudioPort() const {
	QString s = ui.audioPort->text();
	if(s.isEmpty())
		return "5006";

	return s;
}

QString SettingsWindow::getPortArgs() const {
	if(isDefault())
		return "";

	return "-P " + getVideoPort() + ":" + getVideoPort()
		+ ":" + getAudioPort() + ":" + getAudioPort() + " ";
}

bool SettingsWindow::isDefault() const {
	return ui.videoPort->text().isEmpty() && ui.audioPort->text().isEmpty();
}
