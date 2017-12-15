#include <QApplication>
#include <QString>
#include <QStringList>
#include <QProcess>
#include <QCoreApplication>
#include <QMessageBox>

#include "ultragrid_window.hpp"

int main(int argc, char *argv[]){
	QApplication app(argc, argv);
	QStringList args = QCoreApplication::arguments();
	QString ultragridExecutable;
	QProcess process;

	int index = args.indexOf("--with-uv");
	if(index != -1 && args.size() >= index + 1) {
		//found
		ultragridExecutable = args.at(index + 1);
	} else {
		ultragridExecutable = "uv";
	}

	process.start(ultragridExecutable);
	if(process.waitForStarted(1000) == false) {
		QMessageBox msgBox;
		msgBox.setText(ultragridExecutable + " doesn't seem to be executable.");
		msgBox.setInformativeText("Please install uv (UltraGrid binary) to your system "
				"path or supply a '--with-uv' parameter.\n\n"
				"Please check also if the binary is executable.");
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.exec();
		return 1;
	}

	UltragridWindow uw;
	uw.show();
	return app.exec();
}
