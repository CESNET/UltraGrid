#!/bin/bash -eux

install_svt() {
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-HEVC && cd SVT-HEVC/Build/linux && ./build.sh release && cd Release && make -j $(nproc) && sudo make install || exit 1 )
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-AV1 && cd SVT-AV1 && cd Build && cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release && make -j $(nproc) && sudo make install || exit 1 )
        git apply SVT-HEVC/ffmpeg_plugin/0001*.patch
}

# The NVENC API implies respective driver version (see libavcodec/nvenc.c), thus we use SDK 8.1 to work with a reasonably old driver version (390)
install_nv_codec_headers() {
        git clone -b sdk/8.1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        ( cd nv-codec-headers && make && sudo make install || exit 1 )
}

git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg
cd /var/tmp/ffmpeg
sed -i '1,/sliceModeData = 1;/s/sliceModeData = 1;/sliceModeData = 8;/' libavcodec/nvenc.c # only first occurence - for H.264, not HEVC
( git clone --depth 1 -b nasm-2.13.xx https://github.com/sezero/nasm.git && cd nasm && ./autogen.sh && ./configure && make nasm.1 && make ndisasm.1 && make -j $(nproc) && sudo make install || exit 1 )
( git clone --depth 1 http://git.videolan.org/git/x264.git && cd x264 && ./configure --disable-static --enable-shared && make -j $(nproc) && sudo make install || exit 1 )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. &&  cmake --build . --parallel && sudo cmake --install . || exit 1 )
install_nv_codec_headers
install_svt
./configure --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx --enable-libspeex --enable-libmp3lame --enable-libsvthevc --enable-libsvtav1
make -j $(nproc)
