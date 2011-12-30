#include "adavncedwindow.h"
#include <QProcess>
#include <QTextStream>
#include <QMessageBox>
#include <QList>
#include <QDebug>
#include <QStringList>
#include "terminal.h"

AdvancedWindow::AdvancedWindow(UltragridSettings *settings, QWidget *parent) :
        QDialog(parent)
{
    this->settings = settings;
    setupUi(this); // this sets up GUI

    connect( pushButton_ok, SIGNAL( clicked() ), this, SLOT( doOK() ) );
    connect( pushButton_save, SIGNAL( clicked() ), this, SLOT( doSave() ) );
    connect( pushButton_cancel, SIGNAL( clicked() ), this, SLOT( doCancel() ) );
    connect( toolButton_capture_help, SIGNAL( clicked() ), this, SLOT( showCaptureHelp() ) );
    connect( toolButton_display_help, SIGNAL( clicked() ), this, SLOT( showDisplayHelp() ) );

    comboBox_capture->addItem("");
    comboBox_display->addItem("");
    comboBox_audio_cap->addItem("");
    comboBox_audio_play->addItem("");

    QStringList out;
    out = getOptionsForParam(QString("-d help"));
    comboBox_display->addItems(out);

    out = getOptionsForParam(QString("-t help"));
    comboBox_capture->addItems(out);

    out = getOptionsForParam(QString("-r help"));
    foreach ( const QString &item, out) {
        comboBox_audio_play->addItem(item, item.section(' ', 0, 0));
    }

    out = getOptionsForParam(QString("-s help"));
    foreach ( const QString &item, out) {
        comboBox_audio_cap->addItem(item, item.section(' ', 0, 0));
    }

    comboBox_capture->setCurrentIndex(comboBox_capture->findText(settings->getSettings().capture));
    comboBox_display->setCurrentIndex(comboBox_display->findText(settings->getSettings().display));
    comboBox_audio_cap->setCurrentIndex(comboBox_audio_cap->findData(settings->getSettings().audio_cap));
    comboBox_audio_play->setCurrentIndex(comboBox_audio_play->findData(settings->getSettings().audio_play));
    lineEdit_display_details->insert(settings->getSettings().display_details);
    lineEdit_capture_details->insert(settings->getSettings().capture_details);
    lineEdit_mtu->insert(settings->getSettings().mtu);
    lineEdit_other->insert(settings->getSettings().other);
}

void AdvancedWindow::saveSettings()
{
    Settings cur;
    cur.mtu = lineEdit_mtu->text();
    cur.display = comboBox_display->currentText();
    cur.capture = comboBox_capture->currentText();
    cur.audio_cap = comboBox_audio_cap->itemData(comboBox_audio_cap->currentIndex()).toString();
    cur.audio_play = comboBox_audio_play->itemData(comboBox_audio_play->currentIndex()).toString();
    cur.display_details = lineEdit_display_details->text();
    cur.capture_details = lineEdit_capture_details->text();
    cur.other = lineEdit_other->text();
    settings->setSettings(cur);
}

void AdvancedWindow::doOK()
{
    saveSettings();
    this->close();
}

void AdvancedWindow::doSave()
{
    saveSettings();
    settings->save();
}

void AdvancedWindow::doCancel()
{
   this->close();
}

 QStringList AdvancedWindow::getOptionsForParam(QString param)
 {
     QStringList out;

     QProcess process;
     QString command("uv ");
     command += param;

     process.start(command);

     process.waitForFinished();
     QByteArray output = process.readAllStandardOutput();
     QList<QByteArray> lines = output.split('\n');

     foreach ( const QByteArray &line, lines ) {
         if(line.size() > 0 && QChar(line[0]).isSpace()) {
             out.append(QString(line).trimmed());
         }
     }

     return out;
 }

 void AdvancedWindow::showCaptureHelp()
 {
     QProcess process;
     QString command("uv -t ");
     command += comboBox_capture->currentText() + ":help";

     process.start(command);

     process.waitForFinished();
     QByteArray output = process.readAllStandardOutput();

     Terminal *term = new Terminal(this);
     term->insertText(output);
     term->show();
 }

 void AdvancedWindow::showDisplayHelp()
 {
     QProcess process;
     QString command("uv -d ");
     command += comboBox_display->currentText() + ":help";

     process.start(command);

     process.waitForFinished();
     QByteArray output = process.readAllStandardOutput();

     Terminal *term = new Terminal(this);
     term->insertText(output);
     term->show();
 }
