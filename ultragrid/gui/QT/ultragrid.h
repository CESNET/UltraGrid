#ifndef MYQTAPP_H
#define MYQTAPP_H

#include "ui_UltraGrid.h"
#include <QProcess>
#include <QCompleter>
#include <QSet>
#include <QString>
#include "ultragridsettings.h"
#include "adavncedwindow.h"
#include "terminal.h"

class UltraGridMainWindow : public QMainWindow, private Ui::UGMainWindowDlg
{
    Q_OBJECT

public:
    UltraGridMainWindow(QWidget *parent = 0);

private:
    QProcess process;
    UltragridSettings settings;
    QStatusBar statusBar;
    QLabel *label;

    QCompleter *completer;
    QSet<QString> history;

    QDialog *advanced;
    Terminal *terminal;

    bool started;

public slots:
    void doStart();
    void doAdvanced();
    void doTerminal();
    void about();
    void UGHasFinished( int exitCode, QProcess::ExitStatus exitStatus );
    void outputAvailable();
    void updateHistory();
};

#endif
