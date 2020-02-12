#!/bin/sh -eux

VERSION=${1:?you need to pass Qt version as an argument}
VER_MAJ_MIN=$(echo $VERSION | sed 's/\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/')

cd /tmp
wget --no-verbose https://download.qt.io/archive/qt/$VER_MAJ_MIN/$VERSION/single/qt-everywhere-src-${VERSION}.tar.xz
tar xJf qt-everywhere-src-${VERSION}.tar.xz
cd qt-everywhere-src-${VERSION}
# TDS SQL compilation fails on macOS
./configure -static -release -no-compile-examples -opensource -confirm-license -opengl -no-sql-tds -no-fontconfig -prefix ${2:-/usr/local/qt}
make -j 3
sudo make install
cd
rm -r /tmp/qt-everywhere-src-${VERSION}.tar.xz /tmp/qt-everywhere-src-${VERSION}

