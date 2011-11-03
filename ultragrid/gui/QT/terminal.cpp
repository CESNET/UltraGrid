#include "terminal.h"

Terminal::Terminal(QDialog *parent) :
    QDialog(parent)
{
    setupUi(this); // this sets up GUI
}

void Terminal::insertText(QString text)
{
    textBrowser->append(text);
}
