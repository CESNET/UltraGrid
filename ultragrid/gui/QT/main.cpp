#include <QApplication>

#include "ultragrid.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    UltraGridMainWindow *window = new UltraGridMainWindow;

    window->show();
    return app.exec();
}

