#!/bin/sh -eux

export CPATH=/usr/local/include${CPATH:+":$CPATH"}
EXTRA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib
export LIBRARY_PATH=$EXTRA_LIB_PATH${LIBRARY_PATH:+":$LIBRARY_PATH"}
export LD_LIBRARY_PATH=$EXTRA_LIB_PATH${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
export PATH="/usr/local/bin:$PATH"
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:+":$PKG_CONFIG_PATH"}

ARCH=$(dpkg --print-architecture)
APPNAME=UltraGrid-latest-${ARCH}.AppImage

./autogen.sh --enable-plugins
make -j "$(nproc)"
make check
make distcheck

./data/scripts/Linux-AppImage/create-appimage.sh
mv -- *.AppImage "$APPNAME"

