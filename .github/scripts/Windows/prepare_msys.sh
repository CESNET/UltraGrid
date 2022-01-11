#!/bin/bash

set -ex

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export MSYSTEM_PREFIX=/ucrt64
export PATH=$MSYSTEM_PREFIX/bin:/usr/local/bin:$PATH
export CPATH=/usr/local/include:/usr/include
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:/usr/lib/pkgconfig:$MSYSTEM_PREFIX/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib

CUDA_D=$(ls -d /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/*)
if test -d "$CUDA_D"; then
        export CPATH=$CPATH:$CUDA_D/include
fi

if test -d /c/Program\ Files/NDI; then
        NDI_D=$(ls -d /c/Program\ Files/NDI/*SDK)
        export CPATH=$CPATH:$NDI_D/Include
        export LIBRARY_PATH=$LIBRARY_PATH:$NDI_D/Lib/x64
        cat $NDI_D/Version.txt | sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' | tee /usr/local/include/ndi_version.h
        FEATURES="$FEATURES --enable-ndi"
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

. ~/.bash_profile

PACMAN_INSTALL='pacman -Sy --needed --noconfirm --disable-download-timeout'
# Install MSYS2 packages
MSYS_REPO=mingw-w64-ucrt-x86_64
$PACMAN_INSTALL automake autoconf git make pkgconf ${MSYS_REPO}-toolchain ${MSYS_REPO}-cppunit unzip zip
$PACMAN_INSTALL ${MSYS_REPO}-speexdsp
$PACMAN_INSTALL ${MSYS_REPO}-glew ${MSYS_REPO}-SDL2 ${MSYS_REPO}-freeglut
$PACMAN_INSTALL ${MSYS_REPO}-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
$PACMAN_INSTALL ${MSYS_REPO}-glib2 ${MSYS_REPO}-curl # RTSP capture
pacman -Scc --noconfirm # make some free space
$PACMAN_INSTALL ${MSYS_REPO}-qt5
$PACMAN_INSTALL ${MSYS_REPO}-imagemagick ${MSYS_REPO}-opencv
$PACMAN_INSTALL p7zip
$PACMAN_INSTALL libtool # PCP
pacman -Scc --noconfirm

# Build AJA wrapper if we have SDK
if test -d /c/AJA; then
        FEATURES="$FEATURES --enable-aja"
        echo "FEATURES=$FEATURES" >> $GITHUB_ENV
        data/scripts/build_aja_lib_win64.sh
fi

# DELTACAST
if [ -n "$SDK_URL" ]; then
        mkdir VideoMaster
        cd VideoMaster
        if curl -f -S $SDK_URL/VideoMaster_SDK_Windows.zip -O; then
                FEATURES="$FEATURES --enable-deltacast"
                echo "FEATURES=$FEATURES" >> $GITHUB_ENV
                unzip VideoMaster_SDK_Windows.zip
                cp Binaries/Resources/Lib64/*dll /usr/local/bin
                cp -r Include/* /usr/local/include
                cp Library/x64/* /usr/local/lib
        fi
        cd ..
        rm -rf VideoMaster
fi

build_cineform() {
        (
        git submodule update --init cineform-sdk
        cd cineform-sdk
        cmake -DBUILD_STATIC=false -DBUILD_TOOLS=false -A x64 . # assume "-G 'Visual Studio 16 2019'"
        cmake --build . --config Release --parallel
        cp Release/CFHDCodec.dll /usr/local/bin && cp Release/CFHDCodec.lib /usr/local/lib && cp Common/* /usr/local/include && cp libcineformsdk.pc /usr/local/lib/pkgconfig
        )
}

$GITHUB_WORKSPACE/.github/scripts/Windows/install_natpmp.sh
$GITHUB_WORKSPACE/.github/scripts/Windows/install_spout.sh

# Install FFMPEG
wget --no-verbose https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z && 7z x ffmpeg-release-full-shared.7z && cp -r ffmpeg-*build-shared/{bin,lib,include} /usr/local && rm -rf ffmpeg-* || exit 1

# Install GPUJPEG
( wget --no-verbose https://github.com/CESNET/GPUJPEG/releases/download/continuous/GPUJPEG.zip && unzip GPUJPEG.zip && cp -r GPUJPEG/* /usr/local || exit 1 )

build_cineform

# Install cross-platform deps
$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh

