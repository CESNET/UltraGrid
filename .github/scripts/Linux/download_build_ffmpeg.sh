#!/bin/bash -eux

git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg
cd /var/tmp/ffmpeg
( git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && make && sudo make install )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. && make && sudo make install )
./configure --enable-shared --enable-gpl --enable-libx264 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx
make
