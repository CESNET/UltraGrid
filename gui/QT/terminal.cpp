#include "terminal.h"

Terminal::Terminal(QWidget *parent) :
        QDialog(parent)
{
    setupUi(this); // this sets up GUI
}

void Terminal::insertText(QString text)
{
    textBrowser->append(text);
}
