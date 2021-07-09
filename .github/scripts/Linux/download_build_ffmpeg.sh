#!/bin/bash -eux

install_svt() {
        sudo apt install yasm
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-HEVC && cd SVT-HEVC/Build/linux && ./build.sh release && cd Release && make && sudo make install || exit 1 )
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-AV1 && cd SVT-AV1 && cd Build && cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release && make -j $(nproc) && sudo make install || exit 1 )
        git apply -3 SVT-HEVC/ffmpeg_plugin/master-*.patch
}

rm -rf /var/tmp/ffmpeg
git clone --depth 1000 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg # n4.3 is needed for SVT HEVC patch
cd /var/tmp/ffmpeg
( git clone --depth 1 -b nasm-2.13.xx https://github.com/sezero/nasm.git && cd nasm && ./autogen.sh && ./configure && make nasm.1 && make ndisasm.1 && make && sudo make install )
( git clone --depth 1 http://git.videolan.org/git/x264.git && cd x264 && ./configure --disable-static --enable-shared && make && sudo make install )
( git clone -b sdk/8.1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && make && sudo make install )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. && make && sudo make install )
install_svt
./configure --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx --enable-libspeex --enable-libmp3lame --enable-libsvthevc --enable-libsvtav1
make
