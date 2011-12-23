#ifndef ADAVNCEDWINDOW_H
#define ADAVNCEDWINDOW_H

#include <QVarLengthArray>
#include "ui_UltraGrid-Advanced.h"
#include "ultragridsettings.h"
#include <QStringList>

class AdvancedWindow : public QDialog, private Ui::AdvancedWindowDlg
{
    Q_OBJECT

public:
    AdvancedWindow(UltragridSettings *settings, QWidget *parent = 0);

private:
    void saveSettings();
    QStringList getOptionsForParam(QString param);

    UltragridSettings *settings;

public slots:
    void doOK();
    void doSave();
    void doCancel();

    void showCaptureHelp();
    void showDisplayHelp();
};


#endif // ADAVNCEDWINDOW_H
