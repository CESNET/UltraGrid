CONFIG += console
HEADERS     = ultragrid.h ultragridsettings.h \
    terminal.h \
    adavncedwindow.h
SOURCES	    = terminal.cpp ultragrid.cpp ultragridsettings.cpp main.cpp \
    adavncedwindow.cpp
FORMS       = Terminal.ui UltraGrid.ui Ultragrid-Advanced.ui

# install
target.path = /usr/local/bin
sources.files = $$SOURCES $$HEADERS $$RESOURCES $$FORMS *.pro
sources.path = /usr/local/src/uv-qt
INSTALLS += target
