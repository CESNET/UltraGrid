#!/bin/sh -eux

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export MSYSTEM_PREFIX=/clang64
export PATH=$MSYSTEM_PREFIX/bin:/usr/local/bin:$PATH
export CPATH=/usr/local/include:/usr/include
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/lib/pkgconfig:$MSYSTEM_PREFIX/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib
export INCLUDE='C:\msys64\clang64\include' # for MSVC (CUDA)

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

github_workspace_cp=$(cygpath "$GITHUB_WORKSPACE")

PACMAN_INSTALL='pacman -Sy --needed --noconfirm --disable-download-timeout'
# Install MSYS2 packages
MINGW_PACKAGE_PREFIX=mingw-w64-clang-x86_64
m=$MINGW_PACKAGE_PREFIX
$PACMAN_INSTALL automake autoconf git make pkgconf $m-toolchain unzip zip
$PACMAN_INSTALL $m-asciidoc \
        $m-ffmpeg \
        $m-libnatpmp \
        $m-vulkan-headers $m-vulkan-loader \

$PACMAN_INSTALL $m-libsoxr $m-speexdsp
$PACMAN_INSTALL $m-glew $m-libcaca $m-SDL2 $m-SDL2_mixer $m-SDL2_ttf $m-glfw
$PACMAN_INSTALL $m-glm
$PACMAN_INSTALL $m-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
$PACMAN_INSTALL $m-curl # RTSP capture
pacman -Scc --noconfirm # make some free space
$PACMAN_INSTALL $m-qt6-base $m-qt6-tools
$PACMAN_INSTALL $m-imagemagick $m-opencv
$PACMAN_INSTALL libtool # PCP
pacman -Scc --noconfirm

# Build AJA wrapper if we have SDK
install_aja() {(
        git clone --depth 1 https://github.com/aja-video/ntv2 AJA
        cd AJA
        "$github_workspace_cp/.github/scripts/download-gh-asset.sh" \
                aja-video/ntv2 libs_windows_ aja_build.zip
        rm README.md # would be overriden from zip below
        unzip aja_build.zip
        mkdir -p lib
        cp Release/*.lib lib/
        cp Release/*.dll /usr/local/bin/
        cd ..
        data/scripts/build_aja_lib_win64.sh
)}

install_deltacast() {(
        if [ -z "$SDK_URL" ]; then
                        return
        fi
        mkdir VideoMaster
        cd VideoMaster
        if curl -f -S "$SDK_URL/VideoMaster_SDK_Windows.zip" -O; then
                FEATURES="$FEATURES --enable-deltacast"
                echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
                unzip VideoMaster_SDK_Windows.zip
                cp Binaries/Resources/Lib64/*dll /usr/local/bin/
                cp -r Include/* /usr/local/include/
                cp Library/x64/* /usr/local/lib/
        fi
        cd ..
        rm -rf VideoMaster
)}

install_gpujpeg() {(
        wget --no-verbose \
https://github.com/CESNET/GPUJPEG/releases/download/continuous/GPUJPEG.zip
        unzip GPUJPEG.zip
        cp -r GPUJPEG/* /usr/local/
)}

install_soundfont() {
        sf_dir="$GITHUB_WORKSPACE/data/Windows/share/soundfonts"
        mkdir -p "$sf_dir"
        cp "$GITHUB_WORKSPACE/data/default.sf3" "$sf_dir"
}

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

install_aja
install_deltacast
install_gpujpeg
install_soundfont

