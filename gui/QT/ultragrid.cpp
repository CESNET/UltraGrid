#include <QtGui> 
#include <QStringList>
#include <QCoreApplication>
#include "ultragrid.h"
#include "ultragridsettings.h"

#include <iostream>

// including <QtGui> saves us to include every class user, <QString>, <QFileDialog>,...

UltraGridMainWindow::UltraGridMainWindow(QWidget *parent)
{
    setupUi(this); // this sets up GUI
    started = false;

    QStringList args = QCoreApplication::arguments();

    int index = args.indexOf("--with-uv");
    if(index != -1 && args.size() >= index + 1) {
        //found
        ultragridExecutable = args.at(index + 1);
    } else {
        ultragridExecutable = "uv";
    }

    history = settings.getHistory();
    completer = new QCompleter(QStringList::fromSet(history), this);

    address->setCompleter(completer);
    address->completer()->setCompletionMode(QCompleter::PopupCompletion);

    // signals/slots
    connect( pushButton_start, SIGNAL( clicked() ), this, SLOT( doStart() ) );
    connect( pushButton_advanced, SIGNAL( clicked() ), this, SLOT( doAdvanced() ) );
    connect( pushButton_terminal, SIGNAL( clicked() ), this, SLOT( doTerminal() ) );
    connect( actionAbout, SIGNAL( triggered() ), this, SLOT( about() ) );
    connect( &process, SIGNAL( finished( int, QProcess::ExitStatus ) ),
             this, SLOT( UGHasFinished( int, QProcess::ExitStatus ) ) );
    connect( &process, SIGNAL( started( ) ),
             this, SLOT( UGHasStarted( ) ) );

    connect( &process, SIGNAL( readyReadStandardOutput () ),
             this, SLOT( outputAvailable ()) );
    connect( &process, SIGNAL( readyReadStandardError () ),
             this, SLOT( outputAvailable ()) );

    setStatusBar(&statusBar);

    terminal = new Terminal(this);
}

void UltraGridMainWindow::doStart()
{
    if(!started)
    {
        QHash<QString, QString> sett = settings.getSettings();
        QString command(ultragridExecutable);
        command += " ";

        if(sett.find("audio_cap") != sett.end() && !sett.find("audio_cap").value().isEmpty())
            command += "-s " + sett.find("audio_cap").value();
        command += + " ";
        if(sett.find("audio_play") != sett.end() && !sett.find("audio_play").value().isEmpty())
            command += "-r " + sett.find("audio_play").value();
        command += + " ";
        if(sett.find("display") != sett.end() && !sett.find("display").value().isEmpty()) {
            command += "-d " + sett.find("display").value();
            if(sett.find("display_details") != sett.end() && !sett.find("display_details").value().isEmpty())
                command +=  ":" + sett.find("display_details").value();
        }
        command += + " ";
        if(sett.find("capture") != sett.end() && !sett.find("capture").value().isEmpty()) {
            command += "-t " + sett.find("capture").value();
            if(sett.find("capture_details") != sett.end() && !sett.find("capture_details").value().isEmpty())
                command +=  ":" + sett.find("capture_details").value();
        }
        command += + " ";
        if(sett.find("mtu") != sett.end() && !sett.find("mtu").value().isEmpty())
            command += "-m " + sett.find("mtu").value();
        command += + " ";
        if(sett.find("compress") != sett.end() && !sett.find("compress").value().isEmpty()) {
            command += "-c " + sett.find("compress").value();
            if(sett.find("compress").value().compare("JPEG") == 0 &&
                    sett.find("compress_jpeg_quality") != sett.end() && !sett.find("compress_jpeg_quality").value().isEmpty()) {
                command += ":" + sett.find("compress_jpeg_quality").value();
            }
        }
        command += + " ";
        if(sett.find("other") != sett.end() && !sett.find("other").value().isEmpty())
            command += sett.find("other").value();
        command += + " ";

        history.insert(address->text());
        updateHistory();
        command += address->text();

        process.setProcessChannelMode(QProcess::MergedChannels);
        process.start(command);
    }
    else
    {
        process.kill();
    }
}


void UltraGridMainWindow::UGHasFinished( int exitCode, QProcess::ExitStatus exitStatus )
{
    pushButton_start->setText("Start");
    if(exitCode == 0)
        statusBar.showMessage("Finished (exited noramlly)");
    else
        statusBar.showMessage("Finished (crashed)");
    started = false;
}

void UltraGridMainWindow::UGHasStarted()
{
    pushButton_start->setText("Stop");
    started = true;
    statusBar.showMessage("Running");
}

void UltraGridMainWindow::doAdvanced()
{
    statusBar.showMessage("please wait few seconds");
    advanced = new AdvancedWindow(&settings, this);

    statusBar.clearMessage();

    advanced->show();
}

void UltraGridMainWindow::doTerminal()
{
    terminal->show();
}


void UltraGridMainWindow::about()
{
    QMessageBox::about(this,"About UltraGrid",
                       "UltraGrid from CESNET is a software "
                       "implementation of high-quality low-latency video and audio transmissions using commodity PC and Mac hardware.\n\n"
                       "More information can be found at http://ultragrid.sitola.cz\n\n"
                       "Please read documents distributed with the product to find out current and former developers."
                       );
}

void UltraGridMainWindow::outputAvailable()
{
    terminal->insertText(process.readAll());
}

void UltraGridMainWindow::updateHistory()
{
    completer = new QCompleter(QStringList::fromSet(history), this);
    completer->setCompletionMode(QCompleter::PopupCompletion);
    address->setCompleter(completer);
    settings.saveHistory(history);
}
