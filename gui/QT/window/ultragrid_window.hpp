#ifndef ULTRAGRID_WINDOW_HPP
#define ULTRAGRID_WINDOW_HPP

#include <QProcess>
#include <vector>
#include <memory>

#include "ui_ultragrid_window.h"
#include "log_window.hpp"
#include "settings_window.hpp"
#include "available_settings.hpp"
#include "settings.hpp"
#include "settings_ui.hpp"

class UltragridWindow : public QMainWindow{
	Q_OBJECT

public:
	UltragridWindow(QWidget *parent = 0);
	static QString findUltragridExecutable();

protected:
	void closeEvent(QCloseEvent *);

private:
	void checkPreview();
	void setupPreviewCallbacks();

	void connectSignals();

	Ui::UltragridWindow ui;

	QString ultragridExecutable;
	QProcess process;
	QProcess previewProcess;

	AvailableSettings availableSettings;

	QString launchArgs;
	QStringList getOptionsForParam(QString param);
	LogWindow log;
	SettingsWindow settingsWindow;

	QTimer previewTimer;
	Settings settings;
	SettingsUi settingsUi;


public slots:
	void about();
	void outputAvailable();
	void start();

	void editArgs(const QString &text);
	void setArgs();

	void showLog();
	void showSettings();

	void startPreview();
	void stopPreview();

private slots:
	void setStartBtnText(QProcess::ProcessState);
	void processStateChanged(QProcess::ProcessState);
	void enablePreview(bool);

	void schedulePreview();
	void refresh();
};


#endif
