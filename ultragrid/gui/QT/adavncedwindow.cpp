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
    connect( compress_JPEG, SIGNAL( toggled(bool) ), compress_JPEG_quality, SLOT( setEnabled(bool)) );

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

    comboBox_capture->setCurrentIndex(comboBox_capture->findText(settings->getValue("capture")));
    comboBox_display->setCurrentIndex(comboBox_display->findText(settings->getValue("display")));
    comboBox_audio_cap->setCurrentIndex(comboBox_audio_cap->findData(settings->getValue("audio_cap")));
    comboBox_audio_play->setCurrentIndex(comboBox_audio_play->findData(settings->getValue("audio_play")));
    lineEdit_display_details->insert(settings->getValue("display_details"));
    lineEdit_capture_details->insert(settings->getValue("capture_details"));

    bool ok;
    QString mtu = settings->getValue("mtu");
    int MTUInt = mtu.toInt(&ok);
    if(ok) {
        spin_mtu->setValue(MTUInt);
    }

    lineEdit_other->insert(settings->getValue("other"));

    QString JPEGQuality = settings->getValue("compress_jpeg_quality");
    int JPEGQualityInt = JPEGQuality.toInt(&ok);
    if(ok) {
        compress_JPEG_quality->setValue(JPEGQualityInt);
    }

    QString compress = settings->getValue("compress");
    if(compress == QString("none")) {
        compress_none->setChecked(true);
    } else if(compress == QString("RTDXT:DXT1")) {
        compress_DXT1->setChecked(true);
    } else if(compress == QString("RTDXT:DXT5")) {
        compress_DXT5->setChecked(true);
    } else if(compress.startsWith("JPEG")) {
        compress_JPEG->setChecked(true);
    }
}

void AdvancedWindow::saveSettings()
{
    settings->setValue("mtu", QString(spin_mtu->value()));
    settings->setValue("display", comboBox_display->currentText());
    settings->setValue("capture", comboBox_capture->currentText());
    settings->setValue("audio_cap", comboBox_audio_cap->itemData(comboBox_audio_cap->currentIndex()).toString());
    settings->setValue("audio_play", comboBox_audio_play->itemData(comboBox_audio_play->currentIndex()).toString());
    settings->setValue("display_details", lineEdit_display_details->text());
    settings->setValue("capture_details", lineEdit_capture_details->text());
    settings->setValue("other", lineEdit_other->text());
    settings->setValue("compress_jpeg_quality", QString(compress_JPEG_quality->value()));

    QString compress;

    if(compress_none->isChecked()) {
        compress = QString("none");
    } else if(compress_DXT1->isChecked()) {
        compress = QString("RTDXT:DXT1");
    } else if(compress_DXT5->isChecked()) {
        compress = QString("RTDXT:DXT5");
    } else if(compress_JPEG->isChecked()) {
        compress = QString("JPEG");
    }

    settings->setValue("compress", compress);
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
