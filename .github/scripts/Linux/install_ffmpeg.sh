#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd nasm && sudo make install )
( cd x264 && sudo make install )
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo make install )
sudo make install
sudo ldconfig
