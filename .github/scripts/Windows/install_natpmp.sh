#!/bin/sh -eu

git clone --depth 1 https://github.com/miniupnp/libnatpmp
cd libnatpmp
cmd /c build.bat
cp natpmp.a /usr/local/lib/libnatpmp.a
cp natpmp_declspec.h natpmp.h /usr/local/include
cd -
rm -rf libnatpmp

