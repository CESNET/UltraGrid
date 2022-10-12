#ifndef ULTRAGRID_WINDOW_HPP
#define ULTRAGRID_WINDOW_HPP

#include <QProcess>
#include <QStringList>
#include <vector>
#include <memory>
#include <string_view>

#include "ui_ultragrid_window.h"
#include "log_window.hpp"
#include "settings_window.hpp"
#include "available_settings.hpp"
#include "settings.hpp"
#include "settings_ui.hpp"
#include "ug_process_manager.hpp"
#include "recv_report.hpp"
#include "line_buffer.hpp"

class UltragridWindow : public QMainWindow{
	Q_OBJECT

public:
	UltragridWindow(QWidget *parent = 0);
	static QString findUltragridExecutable();

	void initializeUgOpts();

protected:
	void closeEvent(QCloseEvent *);

private:
	void checkPreview();
	void setupPreviewCallbacks();

	void connectSignals();

	void loadSettingsFile(const QString &filename);

	static void schedulePreview(Option&, bool, void *);

	void processOutputLine(std::string_view line);

	Ui::UltragridWindow ui;

	QString ultragridExecutable;
	UgProcessManager processMngr;

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

	LineBuffer lineBuf;
	RecvReportWidget receiverLoss;

	QLabel versionLabel;


public slots:
	void about();
	void outputAvailable();
	void start();

	void editArgs(const QString &text);
	void setArgs();

	void showLog();
	void showSettings();

	void updatePreview();
	void switchToPreview();
	bool launchPreview();

	void saveSettings();
	void loadSettings();

private slots:
	void processStateChanged(UgProcessManager::State state);
	void unexpectedExit(UgProcessManager::State state,
		int code, QProcess::ExitStatus status);
	void enablePreview(bool);

	void refresh();
};


#endif
