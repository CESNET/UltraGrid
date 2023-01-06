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
#include "launch_manager.hpp"
#include "recv_report.hpp"
#include "recv_loss.hpp"
#include "line_buffer.hpp"
#include "control_port.hpp"

class UltragridWindow : public QMainWindow{
	Q_OBJECT

public:
	UltragridWindow(QWidget *parent = 0);
	static QString findUltragridExecutable();

	void initializeUgOpts();

protected:
	void closeEvent(QCloseEvent *);

private:
	void start();
	void checkPreview();
	void setupPreviewCallbacks();

	void connectSignals();

	void loadSettingsFile(const QString &filename);

	static void schedulePreview(Option&, bool, void *);

	Ui::UltragridWindow ui;

	QString ultragridExecutable;

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

	RecvReportWidget rtcpRr;
	RecvLossWidget receiverLoss;

	QLabel versionLabel;

	ControlPort controlPort;

	LaunchManager launchMngr;

public slots:
	void about();
	void startStop();

	void editArgs(const QString &text);
	void setArgs();

	void showLog();
	void showSettings();

	void updatePreview();
	bool launchPreview();

	void saveSettings();
	void loadSettings();

private slots:
	void enablePreview(bool);

	void refresh();
};


#endif
