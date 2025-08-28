#!/bin/bash -eux

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/common.sh" # for get_build_deps_excl

# build dir that will be restored from cache
cache_dir=/var/tmp/ffmpeg

# install the deps - runs always (regardless the cache)
deps() {
        ffmpeg_build_dep=$(get_build_deps_excl ffmpeg 'libsdl')
        # shellcheck disable=SC2086 # intentional
        sudo apt install $ffmpeg_build_dep libdav1d-dev libde265-dev \
                libopenh264-dev
        sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' \
                libvpx-dev nginx
}

install_aom() {(
        git clone --depth 1 https://aomedia.googlesource.com/aom
        mkdir -p aom/build
        cd aom/build
        cmake -DBUILD_SHARED_LIBS=1 ..
        cmake --build . --parallel "$(nproc)"
        sudo cmake --install .
)}

install_libvpx() {(
        git clone --depth 1 https://github.com/webmproject/libvpx.git
        cd libvpx
        ./configure --enable-pic --disable-examples --disable-install-bins \
                        --disable-install-srcs --enable-vp9-highbitdepth
        make -j "$(nproc)"
        sudo make install
)}

install_svt() {
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-HEVC &&
                cd SVT-HEVC/Build/linux && ./build.sh release && cd Release &&
                sudo cmake --install . || exit 1 )
        ( git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git &&
                cd SVT-AV1 && cd Build &&
                cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release &&
                cmake --build . --parallel && sudo cmake --install . || exit 1 )
        ( git clone --depth 1 https://github.com/OpenVisualCloud/SVT-VP9.git &&
                cd SVT-VP9/Build && cmake .. -DCMAKE_BUILD_TYPE=Release &&
                cmake --build . --parallel && sudo cmake --install . || exit 1 )
        # libsvtav1 in FFmpeg upstream, for SVT-HEVC now our custom patch in ffmpeg-patches
        # if patch apply fails, try increasing $FFMPEG_GIT_DEPTH
        git am -3 SVT-VP9/ffmpeg_plugin/master-*.patch
        # TOD TOREMOVE when not needed
        sed 's/\* avctx->ticks_per_frame//' libavcodec/libsvt_vp9.c >fix
        mv fix libavcodec/libsvt_vp9.c
}

# The NV Video Codec SDK headers version 12.0 implies driver v520.56.06 in Linux
install_nv_codec_headers() {
        git clone --depth 1 -b sdk/12.0 https://github.com/FFmpeg/nv-codec-headers
        ( cd nv-codec-headers && make && sudo make install || exit 1 )
}

install_onevpl() {(
        git clone --depth 1 https://github.com/oneapi-src/oneVPL
        mkdir oneVPL/build
        cd oneVPL/build
        cmake -DBUILD_TOOLS=OFF ..
        cmake --build . --config Release --parallel
        sudo cmake --build . --config Release --target install
)}

# build FFmpeg deps + FFmpeg itself
build_install() {
        rm -rf $cache_dir
        FFMPEG_GIT_DEPTH=5000 # greater depth is useful for 3-way merges
        git clone --depth $FFMPEG_GIT_DEPTH https://github.com/FFmpeg/FFmpeg.git \
                $cache_dir
        cd $cache_dir
        install_aom
        install_libvpx
        install_nv_codec_headers
        install_onevpl
        install_svt
        # apply patches
        find "$GITHUB_WORKSPACE/.github/scripts/Linux/ffmpeg-patches" \
                -name '*.patch' -print0 | sort -z | xargs -0 -n 1 git am -3
        ./configure --disable-static --enable-shared --enable-gpl --enable-nonfree \
                --disable-sdl2 \
                --enable-libaom \
                --enable-libdav1d \
                --enable-libde265 \
                --enable-libmp3lame \
                --enable-libopenh264 \
                --enable-libopus \
                --enable-librav1e \
                --enable-libspeex \
                --enable-libsvtav1 \
                --enable-libsvthevc \
                --enable-libsvtvp9 \
                --enable-libvpl \
                --enable-libvpx \
                --enable-libx264 \
                --enable-libx265 \
                --enable-nvenc \
                --enable-vulkan \

        make -j "$(nproc)"
        sudo make install
        sudo ldconfig
}

# if cache is successfully restored, just install the builds
install_cached() {
        cd $cache_dir
        ( cd libvpx && sudo make install )
        ( cd nv-codec-headers && sudo make install )
        ( cd aom/build && sudo cmake --install . )
        sudo cmake --install SVT-AV1/Build
        sudo cmake --install SVT-HEVC/Build/linux/Release
        sudo cmake --install SVT-VP9/Build
        sudo cmake --build oneVPL/build --config Release --target install

        sudo make install
        sudo ldconfig
}

deps
if [ -d $cache_dir ]; then
        install_cached
else
        build_install
fi
