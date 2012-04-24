#include <QtGui> 
#include <QStringList>
#include "ultragrid.h"
#include "ultragridsettings.h"

// including <QtGui> saves us to include every class user, <QString>, <QFileDialog>,...

UltraGridMainWindow::UltraGridMainWindow(QWidget *parent)
{
    setupUi(this); // this sets up GUI
    started = false;

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
        pushButton_start->setText("Stop");
        started = true;
        statusBar.showMessage("Running");
        QHash<QString, QString> sett = settings.getSettings();
        QString command("uv ");
        if(sett.find("audio_cap") != sett.end())
            command += "-s " + sett.find("audio_cap").value();
        command += + " ";
        if(sett.find("audio_play") != sett.end())
            command += "-r " + sett.find("audio_play").value();
        command += + " ";
        if(sett.find("display") != sett.end()) {
            command += "-d " + sett.find("display").value();
            if(sett.find("display_details") != sett.end())
                command +=  ":" + sett.find("display_details").value();
        }
        command += + " ";
        if(sett.find("capture") != sett.end()) {
            command += "-t " + sett.find("capture").value();
            if(sett.find("capture_details") != sett.end())
                command +=  ":" + sett.find("capture_details").value();
        }
        command += + " ";
        if(sett.find("mtu") != sett.end())
            command += "-m " + sett.find("mtu").value();
        command += + " ";
        if(sett.find("compress") != sett.end()) {
            command += sett.find("compress").value();
            if(sett.find("compress").value().compare("JPEG") == 0 &&
                    sett.find("compress_jpeg_quality") != sett.end()) {
                command += ":" + sett.find("compress_jpeg_quality").value();
            }
        }
        command += + " ";
        if(sett.find("other") != sett.end())
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

void UltraGridMainWindow::doAdvanced()
{
    advanced = new AdvancedWindow(&settings, this);

    advanced->show();
}

void UltraGridMainWindow::doTerminal()
{
    terminal->show();
}


void UltraGridMainWindow::about()
{
    QMessageBox::about(this,"About myQtApp",
                       "This app was coded for educational purposes.\n"
                       "Number 1 is: "
                       "Bye.\n");
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
