#!/bin/bash -eux

install_libvpx() {
        (
        git clone --depth 1 https://github.com/webmproject/libvpx.git
        cd libvpx
        ./configure --enable-pic --disable-examples --disable-install-bins --disable-install-srcs --enable-vp9-highbitdepth
        make -j $(nproc)
        sudo make install
        )
}

install_svt() {
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-HEVC && cd SVT-HEVC/Build/linux && ./build.sh release && cd Release && make -j $(nproc) && sudo make install || exit 1 )
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-AV1 && cd SVT-AV1 && cd Build && cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release && make -j $(nproc) && sudo make install || exit 1 )
        git apply SVT-HEVC/ffmpeg_plugin/0001*.patch
}

# The NVENC API implies respective driver version (see libavcodec/nvenc.c), consider capping the version for backward compatibility
install_nv_codec_headers() {
        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        ( cd nv-codec-headers && make && sudo make install || exit 1 )
}

git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg
cd /var/tmp/ffmpeg
( git clone --depth 1 -b nasm-2.13.xx https://github.com/sezero/nasm.git && cd nasm && ./autogen.sh && ./configure && make nasm.1 && make ndisasm.1 && make -j $(nproc) && sudo make install || exit 1 )
( git clone --depth 1 http://git.videolan.org/git/x264.git && cd x264 && ./configure --disable-static --enable-shared && make -j $(nproc) && sudo make install || exit 1 )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. &&  cmake --build . --parallel && sudo cmake --install . || exit 1 )
install_libvpx
install_nv_codec_headers
install_svt
# apply patches
for n in $GITHUB_WORKSPACE/.github/scripts/Linux/ffmpeg-patches/*patch; do
        git apply $n
done
./configure --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx --enable-libspeex --enable-libmp3lame --enable-libsvthevc --enable-libsvtav1
make -j $(nproc)
