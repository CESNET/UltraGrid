#include <QMessageBox>

#include "ultragrid_window.hpp"

UltragridWindow::UltragridWindow(QWidget *parent): QMainWindow(parent){
	ui.setupUi(this);
	ui.terminal->setVisible(false);

	QStringList args = QCoreApplication::arguments();
	int index = args.indexOf("--with-uv");
	if(index != -1 && args.size() >= index + 1) {
		//found
		ultragridExecutable = args.at(index + 1);
	} else {
		ultragridExecutable = "uv";
	}

	connect(ui.actionAbout_UltraGrid, SIGNAL(triggered()), this, SLOT(about()));
	connect(ui.startButton, SIGNAL(clicked()), this, SLOT(start()));
	connect(&process, SIGNAL(readyReadStandardOutput()), this, SLOT(outputAvailable()));
	connect(&process, SIGNAL(readyReadStandardError()), this, SLOT(outputAvailable()));
	connect(&process, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(setStartBtnText(QProcess::ProcessState)));

	connect(ui.networkDestinationEdit, SIGNAL(textEdited(const QString &)), this, SLOT(setArgs()));

	connect(ui.arguments, SIGNAL(textChanged(const QString &)), this, SLOT(editArgs(const QString &)));
	connect(ui.editCheckBox, SIGNAL(toggled(bool)), this, SLOT(setArgs()));
	connect(ui.actionRefresh, SIGNAL(triggered()), this, SLOT(queryOpts()));
	connect(ui.actionAdvanced, SIGNAL(toggled(bool)), this, SLOT(setAdvanced(bool)));
	connect(ui.actionShow_Terminal, SIGNAL(triggered()), this, SLOT(showLog()));

	opts.emplace_back(new SourceOption(&ui,
				ultragridExecutable));

	opts.emplace_back(new DisplayOption(&ui,
				ultragridExecutable));

	opts.emplace_back(new VideoCompressOption(&ui,
				ultragridExecutable));

	opts.emplace_back(new AudioSourceOption(&ui,
				ultragridExecutable));

	opts.emplace_back(new GenericOption(ui.audioPlaybackComboBox,
				ultragridExecutable,
				QString("-r")));

	opts.emplace_back(new AudioCompressOption(&ui,
				ultragridExecutable));

	opts.emplace_back(new FecOption(&ui));

	for(auto &opt : opts){
		connect(opt.get(), SIGNAL(changed()), this, SLOT(setArgs()));
	}

	queryOpts();
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
#if 1
	ui.terminal->moveCursor(QTextCursor::End);
	ui.terminal->insertPlainText(str);
	ui.terminal->moveCursor(QTextCursor::End);
#endif
	log.write(str);
}

void UltragridWindow::start(){
	if(process.pid() > 0){
		process.terminate();
		return;
	}

	QString command(ultragridExecutable);

	command += " ";
	command += launchArgs;
	process.setProcessChannelMode(QProcess::MergedChannels);
	log.write("Command: " + command + "\n\n");
	process.start(command);
}

void UltragridWindow::editArgs(const QString &text){
	launchArgs = text;
}

void UltragridWindow::setArgs(){
	launchArgs = "";

	for(auto &opt : opts){
		launchArgs += opt->getLaunchParam();
	}

	launchArgs += ui.networkDestinationEdit->text();

	ui.arguments->setText(launchArgs);
}

void UltragridWindow::queryOpts(){
	for(auto &opt : opts){
		opt->queryAvailOpts();
	}

	setArgs();
}

void UltragridWindow::setAdvanced(bool adv){
	for(auto &opt : opts){
		opt->setAdvanced(adv);
	}
	queryOpts();
}

void UltragridWindow::showLog(){
	log.show();
	log.raise();
}

void UltragridWindow::setStartBtnText(QProcess::ProcessState s){
	if(s == QProcess::Running){
		ui.startButton->setText("Stop");
	} else {
		ui.startButton->setText("Start");
	}
}
