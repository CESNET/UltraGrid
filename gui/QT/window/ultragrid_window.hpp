#ifndef ULTRAGRID_WINDOW_HPP
#define ULTRAGRID_WINDOW_HPP

#include <QProcess>
#include <QStringList>
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

	void loadSettingsFile(const QString &filename);

	static void schedulePreview(Option&, bool, void *);

	Ui::UltragridWindow ui;

	QString ultragridExecutable;
	QProcess process;
	QProcess previewProcess;

	AvailableSettings availableSettings;

	QStringList launchArgs;
	QStringList getOptionsForParam(QString param);
	LogWindow log;
	SettingsWindow settingsWindow;

	QTimer previewTimer;
	Settings settings;
	SettingsUi settingsUi;

	QLabel processStatus;
	QLabel previewStatus;

	QLabel versionLabel;


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

	void saveSettings();
	void loadSettings();

private slots:
	void setStartBtnText(QProcess::ProcessState);
	void processStateChanged(QProcess::ProcessState);
	void processFinished(int, QProcess::ExitStatus);
	void previewStateChanged(QProcess::ProcessState);
	void previewFinished(int, QProcess::ExitStatus);
	void enablePreview(bool);

	void refresh();
};


#endif
