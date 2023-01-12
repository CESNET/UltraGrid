#!/bin/sh -eux

git clone --depth 1 https://github.com/miniupnp/libnatpmp
cd libnatpmp

gcc -c -Wall -Os -DWIN32 -DNATPMP_STATICLIB -DENABLE_STRNATPMPERR getgateway.c
gcc -c -Wall -Os -DWIN32 -DNATPMP_STATICLIB -DENABLE_STRNATPMPERR natpmp.c
gcc -c -Wall -Os -DWIN32 wingettimeofday.c
ar cr natpmp.a getgateway.o natpmp.o wingettimeofday.o

cp natpmp.a /usr/local/lib/libnatpmp.a
cp natpmp_declspec.h natpmp.h /usr/local/include
cd -
rm -rf libnatpmp

