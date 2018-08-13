#include <QClipboard>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include "log_window.hpp"

LogWindow::LogWindow(QWidget *parent): QDialog(parent){
	ui.setupUi(this);

	connect(ui.copyBtn, SIGNAL(clicked()), this, SLOT(copyToClipboard()));
	connect(ui.saveBtn, SIGNAL(clicked()), this, SLOT(saveToFile()));
}

void LogWindow::write(const QString& str){
	ui.terminal->moveCursor(QTextCursor::End);
	ui.terminal->insertPlainText(str);
	ui.terminal->moveCursor(QTextCursor::End);
}

void LogWindow::copyToClipboard(){
	QClipboard *clip = QApplication::clipboard();

	QString str = ui.terminal->toPlainText();

	clip->setText(str);
}

void LogWindow::saveToFile(){
    QString fileName = QFileDialog::getSaveFileName(this,
			tr("Save log"), "",
			tr("Text File (*.txt);;All Files (*)"));	

	if(fileName.isEmpty())
		return;

	QFile file(fileName);

	if(file.open(QIODevice::WriteOnly)){
		QTextStream stream(&file);
		stream << ui.terminal->toPlainText();
	}
}
