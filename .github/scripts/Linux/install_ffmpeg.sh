#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd nasm && sudo make install )
( cd libvpx && sudo make install )
( cd x264 && sudo make install )
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo cmake --install . )
( cd SVT-HEVC/Build/linux/Release && sudo make install )
( cd SVT-AV1/Build && sudo make install )
sudo make install
sudo ldconfig
