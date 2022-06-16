#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd libvpx && sudo make install )
( cd x264 && sudo make install )
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo cmake --install . )
( cd SVT-AV1/Build && sudo make install )
( cd SVT-HEVC/Build/linux/Release && sudo make install || exit 1 )
( cd SVT-VP9/Build && sudo make install || exit 1 )
sudo make install
sudo ldconfig
