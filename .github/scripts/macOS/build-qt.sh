#!/bin/bash

set -ex

wget --no-verbose https://download.qt.io/archive/qt/5.13/5.13.2/single/qt-everywhere-src-5.13.2.tar.xz
tar xJf qt-everywhere-src-5.13.2.tar.xz
cd qt-everywhere-src-5.13.2
./configure -static -release -nomake examples -opensource -confirm-license -opengl -prefix /usr/local/opt/qt
make -j 3 && sudo make install

