#!/bin/sh -eu

curl -L http://miniupnp.free.fr/files/download.php?file=libnatpmp-20150609.tar.gz | tar xz
cd libnatpmp-*
cmd /c build.bat
cp natpmp.a /usr/local/lib/libnatpmp.a
cp declspec.h natpmp.h /usr/local/include
cd -
rm -rf libnatpmp-*

