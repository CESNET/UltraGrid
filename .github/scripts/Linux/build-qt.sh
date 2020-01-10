#!/bin/sh

set -ex

wget --no-verbose https://download.qt.io/archive/qt/5.13/5.13.2/single/qt-everywhere-src-5.13.2.tar.xz -O /var/tmp/qt.tar.xz
cd /var/tmp && tar xJf qt.tar.xz && mv qt-everywhere-src-* qt
cd /var/tmp/qt && ./configure -static -release -nomake examples -opensource -confirm-license -opengl -prefix /usr/local/qt
make -j 3 -C /var/tmp/qt && sudo make -C /var/tmp/qt install

