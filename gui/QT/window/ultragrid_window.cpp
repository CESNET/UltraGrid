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

	previewTimer.setSingleShot(true);
	connect(&previewTimer, SIGNAL(timeout()), this, SLOT(updatePreview()));
	setupPreviewCallbacks();

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

	launchQuery();

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

	QString verString(GIT_CURRENT_SHA1 "/" GIT_CURRENT_BRANCH);

	versionLabel.setText(QString("Ver: ") + verString);

	receiverLoss.reset();

	ui.localPreviewBar->addWidget(&rtcpRr);
	ui.localPreviewBar->setAlignment(&rtcpRr, Qt::AlignBottom);

	ui.remotePreviewBar->addWidget(&receiverLoss);
	ui.remotePreviewBar->setAlignment(&receiverLoss, Qt::AlignBottom);

	ui.statusbar->addPermanentWidget(&versionLabel);

	ui.vuMeter->setControlPort(&controlPort);
	ui.vuMeter->setOnRightSide(true);
	ui.vuMeterSend->setControlPort(&controlPort);
	ui.vuMeterSend->setParsePrefix("stats ASEND");

	using namespace std::placeholders;
	controlPort.addLineCallback(std::bind(&BandwidthWidget::parseLine, ui.send_bandwidth, _1));
}

void UltragridWindow::launchQuery(){
	auto ctx = std::make_unique<LaunchContext>();
	ctx->executablePath = ultragridExecutable;
	ctx->args = QStringList() << "--capabilities";
	ctx->type = LaunchContext::Type::Query;

	struct QueryData : public LaunchContext::ExtraData{
		QLabel queryMsg = QLabel("Querying UltraGrid capabilities...");

		bool safeMode = false;
		std::string currentQuery;

		std::vector<std::string> safeModeQueue;
		std::vector<std::string> safeModeFails;
	};

	ctx->extraData = std::make_unique<QueryData>();
	auto extraPtr = static_cast<QueryData *>(ctx->extraData.get());

	ui.statusbar->addWidget(&extraPtr->queryMsg);
	availableSettings.queryBegin();

	connect(&ctx->process, &QProcess::started, 
			[=]()
			{
				ui.startButton->setEnabled(false);
				ui.startButton->setText("Querying...");
			});

	connect(ctx.get(), &LaunchContext::processTerminated,
			[=](int ec, QProcess::ExitStatus es, bool requested)
			{
				ui.startButton->setEnabled(true);
				ui.startButton->setText("Start");

				if(!requested && (es == QProcess::ExitStatus::CrashExit || ec != 0)){
					if(!extraPtr->safeMode){
						extraPtr->safeMode = true;
						ui.statusbar->showMessage("Capabilities querying in safe mode...");
						for(int i = 0; i < SettingType::SETTING_TYPE_COUNT; i++){
							SettingType type = static_cast<SettingType>(i);
							for(const auto& mod : availableSettings.getAvailableSettings(type)){
								std::string item = settingTypeToStr(type);
								item += ":";
								item += mod;
								extraPtr->safeModeQueue.push_back(std::move(item));	
							}
						}
						extraPtr->safeModeQueue.emplace_back("noprobe");	
						availableSettings.queryBegin();
					} else {
						extraPtr->safeModeFails.push_back(extraPtr->currentQuery);
					}
				}

				if(extraPtr->safeMode && !extraPtr->safeModeQueue.empty()){
					auto next_ctx = launchMngr.extractCurrentCtx();

					auto modToQuery = std::move(extraPtr->safeModeQueue.back());
					extraPtr->safeModeQueue.pop_back();

					extraPtr->currentQuery = "--capabilities=" + modToQuery;

					next_ctx->args = QStringList()
						<< QString::fromStdString(extraPtr->currentQuery);

					availableSettings.queryBeginPass();
					launchMngr.reLaunch(std::move(next_ctx));

					return;
				}

				if(extraPtr->safeMode && !extraPtr->safeModeFails.empty()){
					QMessageBox msgBox;
					std::string msg = "Ultragrid crashed when querying the following modules."
							" Some options may be missing as a result.\n"; 

					for(const auto& i : extraPtr->safeModeFails){
						msg += "\n";
						msg += i;
					}
					msgBox.setText(QString::fromStdString(msg));
					msgBox.setIcon(QMessageBox::Warning);
					msgBox.exec();
				}

				availableSettings.queryEnd();
				settings.populateSettingsFromCapabilities(&availableSettings);
				settingsUi.refreshAll();
				checkPreview();
				launchPreview();
			});

	connect(ctx.get(), &LaunchContext::processOutputLine,
			[=](std::string_view line) { availableSettings.queryLine(line); });

	launchMngr.launch(std::move(ctx));
}

void UltragridWindow::refresh(){
	launchQuery();
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
			"UltraGrid uses <a href=\"http://ndi.video\">NDI&#174;</a>, NDI&#174; is a registered trademark of Vizrt NDI AB"
			);
	aboutBox.setStandardButtons(QMessageBox::Ok);
	aboutBox.exec();
}

void UltragridWindow::startStop(){
	if(launchMngr.getCurrentStatus() == LaunchContext::Type::Run){
		ui.startButton->setText("Stopping...");
		processStatus.setText("UG: Stopping");
		ui.startButton->setEnabled(false);
		launchMngr.stop();
	} else {
		start();
	}
}

void UltragridWindow::start(){
	log.write("Command args: " + argListToString(launchArgs) + "\n\n");
	auto ctx = std::make_unique<LaunchContext>();
	ctx->executablePath = ultragridExecutable;
	ctx->args = launchArgs;
	ctx->type = LaunchContext::Type::Run;

	connect(&ctx->process, &QProcess::started,
			[=]()
			{
				ui.startButton->setText("Stop");
				ui.startButton->setEnabled(true);
				processStatus.setText("UG: Running");
				ui.actionRefresh->setEnabled(false);
			});

	connect(ctx.get(), &LaunchContext::processTerminated,
			[=](int ec, QProcess::ExitStatus es, bool requested)
			{
				ui.startButton->setText("Start");
				ui.startButton->setEnabled(true);
				ui.actionRefresh->setEnabled(true);
				receiverLoss.reset();
				rtcpRr.reset();
				ui.send_bandwidth->reset();

				if(!requested && (es == QProcess::ExitStatus::CrashExit || ec != 0)){
					processStatus.setText("UG: Crashed");

					log.write("Process exited with code: " + QString::number(ec) + "\n\n");
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

				} else {
					processStatus.setText("UG: Stopped");
					launchPreview();
				}
			});

	connect(ctx.get(), &LaunchContext::processOutputRead,
			[=](QString str) { log.write(str); });

	connect(ctx.get(), &LaunchContext::processOutputLine,
			[=](std::string_view sv)
			{
				rtcpRr.parseLine(sv);
				controlPort.parseLine(sv);
				receiverLoss.parseLine(sv);
			});

	ui.startButton->setText("Starting...");
	processStatus.setText("UG: Starting");
	ui.startButton->setEnabled(false);
	launchMngr.launch(std::move(ctx));
}

void UltragridWindow::schedulePreview(Option&, bool, void *opaque){
	UltragridWindow *obj = static_cast<UltragridWindow *>(opaque);
	obj->previewTimer.start(200);
}

void UltragridWindow::updatePreview(){
	if(launchMngr.getCurrentStatus() == LaunchContext::Type::Run
			|| launchMngr.getCurrentStatus() == LaunchContext::Type::Query)
	{
		return;
	}

	launchPreview();
}

bool UltragridWindow::launchPreview(){
	if(!ui.previewCheckBox->isEnabled()
			|| !ui.previewCheckBox->isChecked())
	{
		return false;
	}

	auto ctx = std::make_unique<LaunchContext>();
	ctx->args = argStringToList(settings.getPreviewParams());
	ctx->executablePath = ultragridExecutable;
	ctx->type = LaunchContext::Type::Preview;

#ifdef DEBUG
	log.write("Preview: " + argListToString(ctx->args) + "\n\n");
#endif

	connect(&ctx->process, &QProcess::started, 
			[=]() { previewStatus.setText("Preview: Running"); });

	connect(ctx.get(), &LaunchContext::processTerminated,
			[=](int ec, QProcess::ExitStatus es, bool requested)
			{
				if(!requested && (es == QProcess::ExitStatus::CrashExit || ec != 0)){
					previewStatus.setText("Preview: Crashed");
				} else {
					previewStatus.setText("Preview: Stopped");
				}
			});

	connect(ctx.get(), &LaunchContext::processOutputLine,
			[=](std::string_view sv) { controlPort.parseLine(sv); });

	launchMngr.launch(std::move(ctx));
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
	if(launchMngr.getCurrentStatus() == LaunchContext::Type::Run){
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

	launchMngr.stop();

	log.close();
	settingsWindow.close();
}

void UltragridWindow::connectSignals(){
	connect(ui.actionAbout_UltraGrid, SIGNAL(triggered()), this, SLOT(about()));
	connect(ui.startButton, SIGNAL(clicked()), this, SLOT(startStop()));

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
	else{
		if(launchMngr.getCurrentStatus() == LaunchContext::Type::Preview){
			launchMngr.stop();
		}
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
