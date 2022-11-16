#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd libvpx && sudo make install )
( cd x264 && sudo make install )
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo cmake --install . )
sudo cmake --install SVT-AV1/Build
( cd SVT-HEVC/Build/linux/Release && sudo make install || exit 1 )
sudo cmake --install SVT-VP9/Build

sudo make install
sudo ldconfig
