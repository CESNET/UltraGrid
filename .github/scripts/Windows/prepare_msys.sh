#!/bin/sh -eux

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export MSYSTEM_PREFIX=/clang64
export PATH=$MSYSTEM_PREFIX/bin:/usr/local/bin:$PATH
export CPATH=/usr/local/include:/usr/include
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/lib/pkgconfig:$MSYSTEM_PREFIX/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib
export INCLUDE='C:\msys64\clang64\include' # for MSVC (CUDA)
export CUDA_FLAGS="--generate-code arch=compute_35,code=sm_35\
 -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -allow-unsupported-compiler"

CUDA_D=$(ls -d /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/*)
if test -d "$CUDA_D"; then
        export CPATH=$CPATH:$CUDA_D/include
fi

if test -d /c/Program\ Files/NDI; then
        NDI_D=$(ls -d /c/Program\ Files/NDI/*SDK)
        export CPATH=$CPATH:$NDI_D/Include
        export LIBRARY_PATH=$LIBRARY_PATH:$NDI_D/Lib/x64
fi

JACK_D=/c/Program\ Files/JACK2
if test -d "$JACK_D"; then
        export PATH=$PATH:$JACK_D/bin
        export CPATH=$CPATH:$JACK_D/include
        export LIBRARY_PATH=$LIBRARY_PATH:$JACK_D/lib
fi

unset temp tmp # defined by /etc/profile, causes CineForm MSBuild fail (GitHub issue #99)

cd `cygpath $GITHUB_WORKSPACE`

EOF

# shellcheck source=/dev/null
. ~/.bash_profile

# shellcheck source=/dev/null
. .github/scripts/json-common.sh

PACMAN_INSTALL='pacman -Sy --needed --noconfirm --disable-download-timeout'
# Install MSYS2 packages
MINGW_PACKAGE_PREFIX=mingw-w64-clang-x86_64
m=$MINGW_PACKAGE_PREFIX
$PACMAN_INSTALL automake autoconf git make pkgconf \
        $m-clang $m-lld $m-winpthreads \
        $m-gcc-compat \
        unzip zip
$PACMAN_INSTALL $m-asciidoc \
        $m-libcaca\
        $m-ffmpeg \
        $m-fluidsynth\
        $m-glew $m-glfw\
        $m-libnatpmp \
        $m-vulkan-headers $m-vulkan-loader \

$PACMAN_INSTALL $m-libsoxr $m-speexdsp
$PACMAN_INSTALL $m-sdl3 $m-sdl3-ttf
$PACMAN_INSTALL $m-glm
$PACMAN_INSTALL $m-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
$PACMAN_INSTALL $m-curl # RTSP capture
pacman -Scc --noconfirm # make some free space
$PACMAN_INSTALL $m-qt6-base $m-qt6-tools
$PACMAN_INSTALL $m-imagemagick $m-opencv
$PACMAN_INSTALL libtool # PCP
pacman -Scc --noconfirm

build_aja_wrapper() {(
        data/scripts/build_aja_lib_win64.sh
)}

install_deltacast() {(
        if [ -z "$SDK_URL" ]; then
                        return
        fi
        mkdir VideoMaster
        cd VideoMaster
        filename=videomaster-win.x64-dev.zip
        if curl -f -S "$SDK_URL/$filename" -O; then
                FEATURES="$FEATURES --enable-deltacast"
                echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
                unzip "$filename"
                cp resources/lib/*dll /usr/local/bin/
                cp resources/lib/*lib /usr/local/lib/
                cp -r resources/include/* /usr/local/include/
        fi
        cd ..
        rm -rf VideoMaster
)}

install_gpujpeg() {(
        fname=GPUJPEG-Windows.zip
        wget --no-verbose \
https://github.com/CESNET/GPUJPEG/releases/download/continuous/"$fname"
        unzip "./$fname"
        cp -r GPUJPEG/* /usr/local/
)}

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

build_aja_wrapper
install_deltacast
install_gpujpeg

