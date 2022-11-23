#include <QMessageBox>
#include <QCloseEvent>
#include <QOpenGLFunctions_3_3_Core>
#include <QFileDialog>
#include <QJsonObject>
#include <QJsonDocument>
#include <QByteArray>
#include <QtGlobal>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QOpenGLVersionFunctionsFactory>
#endif

#include "ultragrid_window.hpp"
#include "debug.hpp"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace {
QString argListToString(const QStringList& argList){
	return argList.join(" ");
}

QStringList argStringToList(const QString& argString){
	QStringList res;

	QString arg;
	bool escaped = false;
	for(int i = 0; i < argString.size(); i++){
		if(argString[i] == '"'){
			escaped = !escaped;
			continue;
		}
		if(argString[i] == ' ' && !escaped){
			if(!arg.isEmpty()){
				res.append(std::move(arg));
				arg = "";
			}
		} else {
			arg.append(argString[i]);
		}
	}

	if(!arg.isEmpty()){
		res.append(std::move(arg));
	}

	return res;
}

QStringList argStringToList(const std::string& argString){
	return argStringToList(QString::fromStdString(argString));
}

} //anonymous namespace

void UltragridWindow::initializeUgOpts(){
	ultragridExecutable = UltragridWindow::findUltragridExecutable();

	processMngr.setUgExecutable(ultragridExecutable);

	previewTimer.setSingleShot(true);
	connect(&previewTimer, SIGNAL(timeout()), this, SLOT(updatePreview()));
	setupPreviewCallbacks();

	availableSettings.queryAll(ultragridExecutable.toStdString());
	settings.populateSettingsFromCapabilities(&availableSettings);

	settingsUi.init(&settings, &availableSettings);
	settingsUi.initMainWin(&ui);
	settingsWindow.init(&settingsUi);

	connectSignals();

	setArgs();

	ui.displayPreview->setKey("ug_preview_disp_unix" + settings.getSessRndKey());
	ui.displayPreview->start();
	ui.capturePreview->setKey("ug_preview_cap_unix" + settings.getSessRndKey());
	ui.capturePreview->start();

	QStringList args = QCoreApplication::arguments();
	int index = args.indexOf("--config");
	if(index != -1 && args.size() >= index + 1) {
		QString filename = args.at(index + 1);
		loadSettingsFile(filename);
	}

	checkPreview();
	launchPreview();
	setupPreviewCallbacks();

	previewStatus.setText("Preview: Stopped");
	processStatus.setText("UG: Stopped");
	ui.statusbar->addWidget(&processStatus);
	ui.statusbar->addWidget(&previewStatus);
}

UltragridWindow::UltragridWindow(QWidget *parent): QMainWindow(parent){
	ui.setupUi(this);
	//ui.terminal->setVisible(false);
#ifdef DEBUG
	ui.actionTest->setVisible(true);
#endif //DEBUG

#ifndef __linux
	ui.actionUse_hw_acceleration->setVisible(false);
#endif

	QString verString(GIT_CURRENT_SHA1);
	verString += GIT_CURRENT_BRANCH;

	versionLabel.setText(QString("Ver: ") + verString);

	receiverLoss.reset();

	ui.localPreviewBar->addWidget(&rtcpRr);
	ui.localPreviewBar->setAlignment(&rtcpRr, Qt::AlignBottom);

	ui.remotePreviewBar->addWidget(&receiverLoss);
	ui.remotePreviewBar->setAlignment(&receiverLoss, Qt::AlignBottom);

	ui.statusbar->addPermanentWidget(&versionLabel);

	ui.vuMeter->setControlPort(&controlPort);

	using namespace std::placeholders;
	controlPort.addLineCallback(std::bind(&BandwidthWidget::parseLine, ui.send_bandwidth, _1));
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
				Option::Callback(&UltragridWindow::schedulePreview, this));
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
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
		ctx.versionFunctions<QOpenGLFunctions_3_3_Core>();
#else
		QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_3_3_Core>(&ctx);
#endif

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
			"Please read documents distributed with the product to find out current and former developers.<br><br>"
			"UltraGrid uses <a href=\"http://ndi.tv\">NDI&#174;</a>, NDI&#174; is a registered trademark of NewTek, Inc."
			);
	aboutBox.setStandardButtons(QMessageBox::Ok);
	aboutBox.exec();
}

void UltragridWindow::processOutputLine(std::string_view line){
	rtcpRr.parseLine(line);
	controlPort.parseLine(line);
	receiverLoss.parseLine(line);
}

void UltragridWindow::outputAvailable(){
	QString str = processMngr.readUgOut();
	if(!str.isEmpty())
		log.write(str);
	else
		str = processMngr.readPreviewOut(); //TODO: Do something more intelligent


	lineBuf.write(str.toStdString());

	while(lineBuf.hasLine()){
		processOutputLine(lineBuf.peek());
		lineBuf.pop();
	}
}

void UltragridWindow::start(){
	log.write("Command args: " + argListToString(launchArgs) + "\n\n");

	if(processMngr.getState() != UgProcessManager::State::UgRunning)
		processMngr.launchUg(launchArgs);
	else
		switchToPreview();
	
}

void UltragridWindow::schedulePreview(Option&, bool, void *opaque){
	UltragridWindow *obj = static_cast<UltragridWindow *>(opaque);
	obj->previewTimer.start(200);
}

void UltragridWindow::updatePreview(){
	if(processMngr.getState() == UgProcessManager::State::UgRunning ||
			processMngr.getState() == UgProcessManager::State::PreviewToUg)
	{
		return;
	}

	launchPreview();
}

void UltragridWindow::switchToPreview(){
	if(!launchPreview())
	{
		processMngr.stopUg();
	}
}

bool UltragridWindow::launchPreview(){
	if(!ui.previewCheckBox->isEnabled()
			|| !ui.previewCheckBox->isChecked())
	{
		return false;
	}

	QStringList previewArgs = argStringToList(settings.getPreviewParams());

#ifdef DEBUG
	log.write("Preview: " + argListToString(previewArgs) + "\n\n");
#endif

	processMngr.launchPreview(previewArgs);
	return true;
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

	if(processMngr.getState() == UgProcessManager::State::UgRunning){
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
	}

	processMngr.stopAll(true);

	log.close();
	settingsWindow.close();
}

void UltragridWindow::connectSignals(){
	connect(ui.actionAbout_UltraGrid, SIGNAL(triggered()), this, SLOT(about()));
	connect(ui.startButton, SIGNAL(clicked()), this, SLOT(start()));
	connect(&processMngr, SIGNAL(ugOutputAvailable()), this, SLOT(outputAvailable()));
	connect(&processMngr, &UgProcessManager::stateChanged,
			this, &UltragridWindow::processStateChanged);
	connect(&processMngr, &UgProcessManager::unexpectedExit,
			this, &UltragridWindow::unexpectedExit);

	connect(ui.arguments, SIGNAL(textChanged(const QString &)), this, SLOT(editArgs(const QString &)));
	connect(ui.editCheckBox, SIGNAL(toggled(bool)), this, SLOT(setArgs()));
	//connect(ui.actionRefresh, SIGNAL(triggered()), this, SLOT(queryOpts()));
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

void UltragridWindow::enablePreview(bool enable){
	if(enable)
		updatePreview();
	else
		processMngr.stopPreview();
}

void UltragridWindow::processStateChanged(UgProcessManager::State state){
	struct Vals{
		const char *btnText;
		bool btnEnabled;
		const char *previewText;
		const char *ugText;
	} vals = [](UgProcessManager::State s) -> Vals {
		typedef UgProcessManager::State ps;
		switch(s){
		case ps::NotRunning: return {"Start", true, "Preview: Stopped", "UG: Stopped"};
		case ps::PreviewRunning: return {"Start", true, "Preview: Running", "UG: Stopped"};
		case ps::PreviewToUg: return {"Starting...", false, "Preview: Stopping", "UG: Starting"};
		case ps::UgRunning: return {"Stop", true, "Preview: Stopped", "UG: Running"};
		case ps::UgToPreview: return {"Stopping...", false, "Preview: Starting", "UG: Stopping"};
		case ps::PreviewToPreview: return {"Start", true, "Preview: Changing", "UG: Stopped"};
		case ps::StoppingAll: return {"Stopping...", false, "Preview: Stopping", "UG: Stopping"};
		default: assert("Invalid state" && false);
		}
	}(state);
	ui.startButton->setText(vals.btnText);
	ui.startButton->setEnabled(vals.btnEnabled);
	previewStatus.setText(vals.previewText);
	processStatus.setText(vals.ugText);

	/* Currently the preview for audio (vuMeter) sends the audio locally and
	 * the volumes are reported from the recv side of ug. This causes ug to
	 * report recv loss from the local loopback stream, which could confuse
	 * users into believing they are receiving something from the network. For
	 * now we just disable the widget when preview is running. TODO: Remove
	 * when separate vuMeters for send and recv are implemented.
	 */
	receiverLoss.setEnabled(state == UgProcessManager::State::UgRunning);
	ui.send_bandwidth->setEnabled(state == UgProcessManager::State::UgRunning);

	receiverLoss.reset();
	rtcpRr.reset();
	ui.send_bandwidth->reset();
}

void UltragridWindow::unexpectedExit(UgProcessManager::State state,
		int code, QProcess::ExitStatus es)
{
	if(state == UgProcessManager::State::PreviewRunning){
		previewStatus.setText("Preview: Crashed");
		return;
	}

	if(state != UgProcessManager::State::UgRunning)
		return; //Don't care about crashes of terminating processes

	if(code == 0 && es == QProcess::ExitStatus::NormalExit){
		//Ug exit was not initiated by GUI, but exited normally
		switchToPreview();
		return;
	}

	processStatus.setText("UG: Crashed");

	log.write("Process exited with code: " + QString::number(code) + "\n\n");
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
// vim: set noet:
