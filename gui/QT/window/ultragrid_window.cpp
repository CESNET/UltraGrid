#include <QMessageBox>
#include <QCloseEvent>
#include <QOpenGLFunctions_3_3_Core>
#include <QFileDialog>
#include <QJsonObject>
#include <QJsonDocument>
#include <QByteArray>
#include <QtGlobal>

#include "ultragrid_window.hpp"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace {
QString argListToString(const QStringList& argList){
	return argList.join(" ");
}

QStringList argStringToList(const QString& argString){
#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
	return argString.split(" ", QString::SkipEmptyParts);
#else
	return argString.split(" ", Qt::SkipEmptyParts);
#endif
}

QStringList argStringToList(const std::string& argString){
#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
	return QString::fromStdString(argString).split(" ", QString::SkipEmptyParts);
#else
	return QString::fromStdString(argString).split(" ", Qt::SkipEmptyParts);
#endif
}
} //anonymous namespace

UltragridWindow::UltragridWindow(QWidget *parent): QMainWindow(parent){
	ui.setupUi(this);
	//ui.terminal->setVisible(false);
#ifdef DEBUG
	ui.actionTest->setVisible(true);
#endif //DEBUG

	ultragridExecutable = UltragridWindow::findUltragridExecutable();

#ifndef __linux
	ui.actionUse_hw_acceleration->setVisible(false);
#endif

	previewTimer.setSingleShot(true);
	connect(&previewTimer, SIGNAL(timeout()), this, SLOT(startPreview()));

	//connect(sourceOption, SIGNAL(changed()), this, SLOT(schedulePreview()));
	//connect(audioSrcOption, SIGNAL(changed()), this, SLOT(schedulePreview()));

	setupPreviewCallbacks();

	availableSettings.queryAll(ultragridExecutable.toStdString());

	settingsUi.init(&settings, &availableSettings);
	settingsUi.initMainWin(&ui);
	settingsWindow.init(&settingsUi);

	connectSignals();

	setArgs();

	ui.displayPreview->setKey("ultragrid_preview_display");
	ui.displayPreview->start();
	ui.capturePreview->setKey("ultragrid_preview_capture");
	ui.capturePreview->start();

	QStringList args = QCoreApplication::arguments();
	int index = args.indexOf("--config");
	if(index != -1 && args.size() >= index + 1) {
		QString filename = args.at(index + 1);
		loadSettingsFile(filename);
	}

	checkPreview();
	startPreview();
	setupPreviewCallbacks();

	previewStatus.setText("Preview: Stopped");
	processStatus.setText("UG: Stopped");
	ui.statusbar->addWidget(&processStatus);
	ui.statusbar->addWidget(&previewStatus);

	QString verString(GIT_CURRENT_SHA1);
	verString += GIT_CURRENT_BRANCH;

	versionLabel.setText(QString("Ver: ") + verString);
	ui.statusbar->addPermanentWidget(&versionLabel);
}

void UltragridWindow::refresh(){
	availableSettings.queryAll(ultragridExecutable.toStdString());
	settingsUi.refreshAll();
}

void UltragridWindow::setupPreviewCallbacks(){
	const char * const opts[] = {
		"video.source", "audio.source", "audio.source.channels", "encryption"
	};

	for(const auto &i : opts){
		settings.getOption(i).addOnChangeCallback(
				std::bind(&UltragridWindow::schedulePreview, this));
	}
}

void UltragridWindow::checkPreview(){
	bool enable = true;
	if(!availableSettings.isAvailable("multiplier", VIDEO_DISPLAY)
			|| !availableSettings.isAvailable("preview", VIDEO_DISPLAY))
	{
		enable = false;

		QMessageBox warningBox(this);
		warningBox.setWindowTitle("Preview disabled");
		warningBox.setText("Preview is disabled, because UltraGrid was compiled" 
				" without preview and multiplier displays."
				" Please build UltraGrid configured with the --enable-qt flag"
				" to enable preview.");
		warningBox.setStandardButtons(QMessageBox::Ok);
		warningBox.setIcon(QMessageBox::Warning);
		warningBox.exec();
	}

	QOpenGLContext ctx;
	ctx.create();
	QOpenGLFunctions_3_3_Core* funcs =
		ctx.versionFunctions<QOpenGLFunctions_3_3_Core>();
	if (!funcs) {
		enable = false;

		QMessageBox warningBox(this);
		warningBox.setWindowTitle("Preview disabled");
		warningBox.setText("OpenGL 3.3 is not supported, disabling preview.");
		warningBox.setStandardButtons(QMessageBox::Ok);
		warningBox.setIcon(QMessageBox::Warning);
		warningBox.exec();
	}

	if(!enable){
		ui.previewCheckBox->setChecked(false);
		ui.previewCheckBox->setEnabled(false);

		settings.getOption("preview").setValue("f");
	}
}

void UltragridWindow::about(){
	QMessageBox aboutBox(this);
	aboutBox.setWindowTitle("About UltraGrid");
	aboutBox.setTextFormat(Qt::RichText);
	aboutBox.setText( "UltraGrid from CESNET is a software "
			"implementation of high-quality low-latency video and audio transmissions using commodity PC and Mac hardware.<br><br>"
			"More information can be found at <a href='http://www.ultragrid.cz'>http://www.ultragrid.cz</a><br><br>"
			"Please read documents distributed with the product to find out current and former developers."
			);
	aboutBox.setStandardButtons(QMessageBox::Ok);
	aboutBox.exec();
}

void UltragridWindow::outputAvailable(){
	//ui.terminal->append(process.readAll());

	QString str = process.readAll();
#if 0
	ui.terminal->moveCursor(QTextCursor::End);
	ui.terminal->insertPlainText(str);
	ui.terminal->moveCursor(QTextCursor::End);
#endif
	log.write(str);
}

void UltragridWindow::start(){
	if(process.processId() > 0){
		process.terminate();
		if(!process.waitForFinished(1000))
			process.kill();
		return;
	}

	stopPreview();

	process.setProcessChannelMode(QProcess::MergedChannels);
	log.write("Command args: " + argListToString(launchArgs) + "\n\n");
	process.start(ultragridExecutable, launchArgs);
}

void UltragridWindow::schedulePreview(){
	previewTimer.start(1000);
}

void UltragridWindow::startPreview(){
	if(!ui.previewCheckBox->isEnabled()
			|| process.state() != QProcess::NotRunning
			|| !ui.previewCheckBox->isChecked())
	{
		return;
	}

	while(previewProcess.state() != QProcess::NotRunning)
		stopPreview();

	QStringList previewArgs = argStringToList(settings.getPreviewParams());
	/*
	if(sourceOption->getCurrentValue() != "none"){
		//We prevent video from network overriding local sources
		//by listening on port 0
		command += " -P 0:0:0:0 ";
	}
	*/

#ifdef DEBUG
	log.write("Preview: " + argListToString(previewArgs) + "\n\n");
#endif
	previewProcess.start(ultragridExecutable, previewArgs);
}

void UltragridWindow::stopPreview(){
	previewProcess.terminate();
	/* The shared preview memory must be released before a new one
	 * can be created. Here we wait 0.5s to allow the preview process
	 * exit gracefully. If it is still running after that we kill it */
	if(!previewProcess.waitForFinished(500)){
		previewProcess.kill();
	}
}

void UltragridWindow::editArgs(const QString &text){
	launchArgs = argStringToList(text);
}

void UltragridWindow::setArgs(){
	QString args = QString::fromStdString(settings.getLaunchParams());

	launchArgs = argStringToList(args);

#ifdef DEBUG
	log.write("set args: " + args + "\n");
#endif
	ui.arguments->setText(args);
}

void UltragridWindow::closeEvent(QCloseEvent *e){

	if(process.processId() > 0){
		QMessageBox sureMsg;
		sureMsg.setIcon(QMessageBox::Question);
		sureMsg.setText(tr("Are you sure?"));
		sureMsg.setInformativeText(tr("UltraGrid is still running. Are you sure you want to exit?\n"));
		sureMsg.setStandardButtons(QMessageBox::Cancel | QMessageBox::No | QMessageBox::Yes);
		sureMsg.setDefaultButton(QMessageBox::No);

		if(sureMsg.exec() != QMessageBox::Yes){
			e->ignore();
			return;
		}

		disconnect(&process, 0, 0, 0);
		process.terminate();
		if(!process.waitForFinished(1000))
			process.kill();
	}

	stopPreview();

	log.close();
	exit(0);
	settingsWindow.close();
}

void UltragridWindow::connectSignals(){
	connect(ui.actionAbout_UltraGrid, SIGNAL(triggered()), this, SLOT(about()));
	connect(ui.startButton, SIGNAL(clicked()), this, SLOT(start()));
	connect(&process, SIGNAL(readyReadStandardOutput()), this, SLOT(outputAvailable()));
	connect(&process, SIGNAL(readyReadStandardError()), this, SLOT(outputAvailable()));
	connect(&process, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(processStateChanged(QProcess::ProcessState)));
	connect(&process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(processFinished(int, QProcess::ExitStatus)));

	connect(&previewProcess, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(previewStateChanged(QProcess::ProcessState)));
	connect(&previewProcess, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(previewFinished(int, QProcess::ExitStatus)));

	connect(ui.arguments, SIGNAL(textChanged(const QString &)), this, SLOT(editArgs(const QString &)));
	connect(ui.editCheckBox, SIGNAL(toggled(bool)), this, SLOT(setArgs()));
	//connect(ui.actionRefresh, SIGNAL(triggered()), this, SLOT(queryOpts()));
	//connect(ui.actionAdvanced, SIGNAL(toggled(bool)), this, SLOT(setAdvanced(bool)));
	connect(ui.actionShow_Terminal, SIGNAL(triggered()), this, SLOT(showLog()));
	connect(ui.actionSettings, SIGNAL(triggered()), this, SLOT(showSettings()));
	connect(ui.previewCheckBox, SIGNAL(toggled(bool)), this, SLOT(enablePreview(bool)));

	connect(ui.actionRefresh, SIGNAL(triggered()), this, SLOT(refresh()));

	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(ui.actionLoad, SIGNAL(triggered()), this, SLOT(loadSettings()));

	connect(&settingsWindow, SIGNAL(changed()), this, SLOT(setArgs()));
	connect(&settingsUi, SIGNAL(changed()), this, SLOT(setArgs()));

	connect(ui.fecToolBtn, SIGNAL(clicked()), this, SLOT(showSettings()));
	connect(ui.fecToolBtn, SIGNAL(clicked()), &settingsWindow, SLOT(fecTab()));
}

void UltragridWindow::showLog(){
	log.show();
	log.raise();
}

void UltragridWindow::showSettings(){
	settingsWindow.show();
	settingsWindow.raise();
}

void UltragridWindow::saveSettings(){
	const auto &optMap = settings.getOptionMap();

	QString filename = QFileDialog::getSaveFileName(this, "Save Settings", "", "Json (*.json)");
	if(!filename.endsWith(QString(".json"))){
		filename.append(QString(".json"));
	}

	QFile outFile(filename);

	if (!outFile.open(QIODevice::WriteOnly)) {
		qWarning("Couldn't open save file.");
		return;
	}

	QJsonObject jsonObj;

	for(const auto &keyValPair : optMap){
		jsonObj[QString::fromStdString(keyValPair.first)] =
			QString::fromStdString(keyValPair.second->getValue());
	}

	QJsonDocument jsonDoc(jsonObj);

	outFile.write(jsonDoc.toJson());

}

void UltragridWindow::loadSettingsFile(const QString &filename){
	QFile inFile(filename);

	if (!inFile.open(QIODevice::ReadOnly)) {
		qWarning("Couldn't open save file.");
		return;
	}

	QByteArray saveData = inFile.readAll();

	QJsonDocument saveDoc = QJsonDocument::fromJson(saveData);
	QJsonObject saveObj = saveDoc.object();

	for(const QString &key : saveObj.keys()){
		if(saveObj[key].isString()){
			settings.getOption(key.toStdString())
				.setValue(saveObj[key].toString().toStdString(), true);
		}
	}

	settings.changedAll();
	setArgs();
}

void UltragridWindow::loadSettings(){
	QFileDialog fileDialog(this);
	fileDialog.setFileMode(QFileDialog::ExistingFile);
	fileDialog.setNameFilter(tr("Json (*.json)"));

	QStringList fileNames;
	if (fileDialog.exec())
		fileNames = fileDialog.selectedFiles();

	if(fileNames.empty())
		return;

	loadSettingsFile(fileNames.first());

}

void UltragridWindow::setStartBtnText(QProcess::ProcessState s){
	if(s == QProcess::Running){
		ui.startButton->setText("Stop");
	} else {
		ui.startButton->setText("Start");
	}
}

void UltragridWindow::enablePreview(bool enable){
	if(enable)
		startPreview();
	else
		stopPreview();
}

void UltragridWindow::processStateChanged(QProcess::ProcessState s){
	setStartBtnText(s);

	if(s == QProcess::NotRunning){
		startPreview();
	} else if(s == QProcess::Running){
		processStatus.setText("UG: Running");
	}
}

void UltragridWindow::processFinished(int code, QProcess::ExitStatus status){
	log.write("Process exited with code: " + QString::number(code) + "\n\n");
	if(status == QProcess::CrashExit || code != 0){
		QMessageBox msgBox(this);
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.setWindowTitle("UltraGrid error!");
		msgBox.setText("Ultragrid has exited with an error! If you need help, please send an email "
				"to ultragrid-dev@cesnet.cz with log attached.");
		QPushButton *showLogBtn = msgBox.addButton(tr("Show log"), QMessageBox::ActionRole);
		msgBox.addButton(tr("Dismiss"), QMessageBox::RejectRole);
		msgBox.exec();

		if(msgBox.clickedButton() == showLogBtn){
			showLog();
		}

		processStatus.setText("UG: Crashed");
	} else {
		processStatus.setText("UG: Stopped");
	}
}

void UltragridWindow::previewStateChanged(QProcess::ProcessState s){
	if(s == QProcess::Running){
		previewStatus.setText("Preview: Running");
	}
}

void UltragridWindow::previewFinished(int code, QProcess::ExitStatus status){
	if(status == QProcess::CrashExit || code != 0){
		previewStatus.setText("Preview: Crashed");
	} else {
		previewStatus.setText("Preview: Stopped");
	}
}

QString UltragridWindow::findUltragridExecutable() {
	QStringList args = QCoreApplication::arguments();
	int index = args.indexOf("--with-uv");
	if(index != -1 && args.size() >= index + 1) {
		//found
		return args.at(index + 1);
	} else {
#ifdef __APPLE__
		CFBundleRef mainBundle = CFBundleGetMainBundle();
		CFURLRef cfURL = CFBundleCopyBundleURL(mainBundle);
		CFStringRef cfPath = CFURLCopyFileSystemPath(cfURL, kCFURLPOSIXPathStyle);
		CFStringEncoding encodingMethod = CFStringGetSystemEncoding();
		QString bundlePath(CFStringGetCStringPtr(cfPath, encodingMethod));
		CFRelease(cfURL);
		CFRelease(cfPath);
		return bundlePath + "/Contents/MacOS/uv";
#else
		return QString("uv");
#endif
	}
}
