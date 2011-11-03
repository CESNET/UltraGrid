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

    terminal = new Terminal;
}

void UltraGridMainWindow::doStart()
{
    if(!started)
    {
        pushButton_start->setText("Stop");
        started = true;
        statusBar.showMessage("Running");
        Settings sett = settings.getSettings();
        QString command("uv ");
        if(!sett.audio_cap.isEmpty())
            command += "-s " + sett.audio_cap ;
        command += + " ";
        if(!sett.audio_play.isEmpty())
            command += "-r " + sett.audio_play;
        command += + " ";
        if(!sett.display.isEmpty()) {
            command += "-d " + sett.display;
            if(!sett.display_details.isEmpty())
                command +=  ":" + sett.display_details ;
        }
        command += + " ";
        if(!sett.capture.isEmpty()) {
            command += "-t " + sett.capture;
            if(!sett.capture_details.isEmpty())
                command +=  ":" + sett.capture_details ;
        }
        command += + " ";
        if(!sett.mtu.isEmpty())
            command += "-m " + sett.mtu;
        command += + " ";
        if(!sett.other.isEmpty())
            command += sett.other;
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
    advanced = new AdvancedWindow(&settings);

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
