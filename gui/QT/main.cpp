#include <QApplication>
#include <QString>
#include <QStringList>
#include <QProcess>
#include <QCoreApplication>
#include <QMessageBox>
#include <QSurfaceFormat>
#include <clocale>

#include "ultragrid_window.hpp"

int main(int argc, char *argv[]){
	QApplication app(argc, argv);
	QStringList args = QCoreApplication::arguments();
	QString ultragridExecutable = UltragridWindow::findUltragridExecutable();
	QProcess process;

	//important: If this line is removed parsing float numbers for vu meter fails
	std::setlocale(LC_NUMERIC, "C");

	process.start("\"" + ultragridExecutable + "\"");
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

	process.waitForFinished();

	if (process.exitStatus() == QProcess::CrashExit || process.exitCode() != 0) {
		QMessageBox msgBox;
		msgBox.setText("UltraGrid has crashes when invoked without arguments!");
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.exec();
		return 1;
	}

	QSurfaceFormat fmt;
	fmt.setDepthBufferSize(24);
	fmt.setStencilBufferSize(8);
	fmt.setVersion(3, 3);
	fmt.setProfile(QSurfaceFormat::CoreProfile);
	QSurfaceFormat::setDefaultFormat(fmt);

	UltragridWindow uw;
	uw.show();
	return app.exec();
}
