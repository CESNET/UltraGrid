#ifndef TERMINAL_H
#define TERMINAL_H

#include <QDialog>
#include <QWidget>
#include <QString>
#include "ui_Terminal.h"

class Terminal : public QDialog, private Ui::TerminalDlg
{
    Q_OBJECT
public:
    explicit Terminal(QWidget *parent = 0);
    void insertText(QString text);

signals:

public slots:

};

#endif // TERMINAL_H
