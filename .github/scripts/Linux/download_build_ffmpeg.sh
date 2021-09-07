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
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-VP9.git && cd SVT-VP9/Build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j $(nproc) && sudo make install || exit 1 )
        git apply -3 SVT-HEVC/ffmpeg_plugin/master-*.patch
        git apply -3 SVT-VP9/ffmpeg_plugin/master-*.patch
}

# The NVENC API implies respective driver version (see libavcodec/nvenc.c) - 455.28 (Linux) / 456.71 (Windows) for v11.0
install_nv_codec_headers() {
        git clone --depth 1 -b sdk/11.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        ( cd nv-codec-headers && make && sudo make install || exit 1 )
}

rm -rf /var/tmp/ffmpeg
git clone --depth 1000 https://git.ffmpeg.org/ffmpeg.git /var/tmp/ffmpeg # depth 1000 useful for 3-way merges
cd /var/tmp/ffmpeg
( git clone --depth 1 http://git.videolan.org/git/x264.git && cd x264 && ./configure --disable-static --enable-shared && make -j $(nproc) && sudo make install || exit 1 )
( git clone --depth 1 https://aomedia.googlesource.com/aom && mkdir -p aom/build && cd aom/build && cmake -DBUILD_SHARED_LIBS=1 .. &&  cmake --build . --parallel && sudo cmake --install . || exit 1 )
install_libvpx
install_nv_codec_headers
install_svt
# apply patches
FF_PATCH_DIR=$GITHUB_WORKSPACE/.github/scripts/Linux/ffmpeg-patches
for n in `ls $FF_PATCH_DIR`; do
        git apply $FF_PATCH_DIR/$n
done
./configure --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libopus --enable-nonfree --enable-nvenc --enable-libaom --enable-libvpx --enable-libspeex --enable-libmp3lame --enable-libsvthevc --enable-libsvtav1 \
        --enable-libdav1d \
        --enable-librav1e \
        --enable-libsvtvp9 \

make -j $(nproc)
sudo make install
sudo ldconfig
