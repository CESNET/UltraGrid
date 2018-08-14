#include <QIntValidator> 
#include "settings_window.hpp"

SettingsWindow::SettingsWindow(QWidget *parent): QDialog(parent){
	ui.setupUi(this);
	ui.basePort->setValidator(new QIntValidator(0, 65535, this));
	ui.controlPort->setValidator(new QIntValidator(0, 65535, this));
}

void SettingsWindow::init(SettingsUi *settingsUi, Settings *s){
	settingsUi->initSettingsWin(&ui);
	settings = s;

	connect(ui.fecNoneRadio, SIGNAL(toggled(bool)), this, SLOT(changeFecPage()));
	connect(ui.fecMultRadio, SIGNAL(toggled(bool)), this, SLOT(changeFecPage()));
	connect(ui.fecLdgmRadio, SIGNAL(toggled(bool)), this, SLOT(changeFecPage()));
	connect(ui.fecRsRadio, SIGNAL(toggled(bool)), this, SLOT(changeFecPage()));

}

void SettingsWindow::changeFecPage(){
	if(ui.fecMultRadio->isChecked()){
		ui.stackedWidget->setCurrentIndex(0);
	} else if(ui.fecLdgmRadio->isChecked()){
		ui.stackedWidget->setCurrentIndex(2);
	} else if(ui.fecRsRadio->isChecked()){
		ui.stackedWidget->setCurrentIndex(3);
	}
}

void SettingsWindow::fecTab(){
	ui.tabWidget->setCurrentIndex(0);
}

