#!/bin/sh -eux

export CPATH=/usr/local/include${CPATH:+":$CPATH"}
EXTRA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib
export LIBRARY_PATH=$EXTRA_LIB_PATH${LIBRARY_PATH:+":$LIBRARY_PATH"}
export LD_LIBRARY_PATH=$EXTRA_LIB_PATH${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
export PATH="/usr/local/bin:$PATH"
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:+":$PKG_CONFIG_PATH"}

ARCH=$(dpkg --print-architecture)
APPNAME=UltraGrid-latest-${ARCH}.AppImage

# set 64-bit off_t for 32-bit ARM, see the following (POSIX 2024 will use V8):
# <https://pubs.opengroup.org/onlinepubs/9699919799/utilities/c99.html>
# TODO: this is duplicite of AC_SYS_LARGEFILE - figure out how to keep
# that without needing to include the config.h in headers
CENV=$(getconf POSIX_V7_WIDTH_RESTRICTED_ENVS | grep -E 'OFFBIG|OFF64' |
  head -n 1)
CFLAGS=${CFLAGS:+$CFLAGS }$(getconf "${CENV}_CFLAGS")
CXXFLAGS=${CXXFLAGS:+$CXXFLAGS }$(getconf "${CENV}_CFLAGS")
export CFLAGS CXXFLAGS

# shellcheck disable=SC2086 # intentional
set -- $FEATURES
set -- --enable-sdl=2  # use SDL2 (environment.sh sets sdl=3)
set -- "$@" --enable-drm_disp

./autogen.sh "$@"
make -j "$(nproc)"
make check
make distcheck

./data/scripts/Linux-AppImage/create-appimage.sh
mv -- *.AppImage "$APPNAME"

