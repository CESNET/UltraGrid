#!/bin/sh -eux

export CPATH=/usr/local/include${CPATH:+":$CPATH"}
EXTRA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib
export LIBRARY_PATH=$EXTRA_LIB_PATH${LIBRARY_PATH:+":$LIBRARY_PATH"}
export LD_LIBRARY_PATH=$EXTRA_LIB_PATH${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
export PATH="/usr/local/bin:$PATH"
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:+":$PKG_CONFIG_PATH"}

ARCH=$(dpkg --print-architecture)
APPNAME=UltraGrid-latest-${ARCH}.AppImage

set -- --enable-plugins --enable-openssl --enable-soxr --enable-speexdsp                                # general
set -- "$@" --enable-alsa --enable-jack --enable-jack-transport --enable-pipewire                       # audio
# vidcap (+ duplex video)
set -- "$@" --enable-decklink --enable-file --enable-ndi --enable-rtsp \
            --enable-screen --enable-swmix --enable-v4l2 --enable-ximea
set -- "$@" --enable-caca --enable-gl-display --enable-panogl_disp --enable-sdl                         # display
set -- "$@" --enable-libavcodec --enable-rtdxt --enable-libswscale --enable-uyvy                        # compression
set -- "$@" --enable-blank --enable-holepunch --enable-natpmp --enable-pcp --enable-resize --enable-scale --enable-sdp-http --enable-testcard-extras --enable-text --enable-video-mixer --enable-zfec # extras (pp. etc)
if [ "$ARCH" = arm64 ]; then
        set -- "$@" --enable-vulkan
fi

./autogen.sh "$@"
make -j "$(nproc)"
make check
make distcheck

./data/scripts/Linux-AppImage/create-appimage.sh
mv -- *.AppImage "$APPNAME"

