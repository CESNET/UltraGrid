#!/bin/bash -eux

git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg
cd /var/tmp/ffmpeg
( git clone --depth 1 -b nasm-2.13.xx https://github.com/sezero/nasm.git && cd nasm && ./autogen.sh && ./configure && make nasm.1 && make ndisasm.1 && make && sudo make install )
( git clone --depth 1 http://git.videolan.org/git/x264.git && cd x264 && ./configure --disable-static --enable-shared && make && sudo make install )
( git clone -b sdk/8.1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && make && sudo make install )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. && make && sudo make install )
./configure --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx --enable-libspeex --enable-libmp3lame
make
