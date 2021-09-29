#!/bin/bash

set -ex

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export PATH=/mingw64/bin:/usr/local/bin:$PATH
export CPATH=/usr/local/include
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/mingw64/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib

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

. ~/.bash_profile

PACMAN_INSTALL='pacman -Sy --needed --noconfirm --disable-download-timeout'
# Install MSYS2 packages
$PACMAN_INSTALL automake autoconf git make pkg-config mingw-w64-x86_64-toolchain mingw-w64-x86_64-cppunit unzip zip
$PACMAN_INSTALL mingw-w64-x86_64-glew mingw-w64-x86_64-SDL2 mingw-w64-x86_64-freeglut
$PACMAN_INSTALL mingw-w64-x86_64-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
$PACMAN_INSTALL mingw-w64-x86_64-glib2 mingw-w64-x86_64-curl # RTSP capture
pacman -Scc --noconfirm # make some free space
$PACMAN_INSTALL mingw-w64-x86_64-qt5
$PACMAN_INSTALL mingw-w64-x86_64-imagemagick mingw-w64-x86_64-opencv
$PACMAN_INSTALL p7zip
pacman -Scc --noconfirm

# Build AJA wrapper if we have SDK
if test -d /c/AJA; then
        data/scripts/build_aja_lib_win64.sh
fi

# DELTACAST
if [ -n "$SDK_URL" ]; then
        mkdir VideoMaster
        cd VideoMaster
        curl -S $SDK_URL/VideoMaster_SDK_Windows.zip -O
        unzip VideoMaster_SDK_Windows.zip
        cp Binaries/Lib64/*dll /usr/local/bin
        cp -r Include/* /usr/local/include
        cp Library/x64/* /usr/local/lib
        cd ..
        rm -rf VideoMaster
fi

build_cineform() {
        (
        git submodule update --init cineform-sdk
        cd cineform-sdk
        cmake -DBUILD_STATIC=false -DBUILD_TOOLS=false -A x64 -G 'Visual Studio 16 2019' .
        cmake --build . --config Release --parallel
        cp Release/CFHDCodec.dll /usr/local/bin && cp Release/CFHDCodec.lib /usr/local/lib && cp Common/* /usr/local/include && cp libcineformsdk.pc /usr/local/lib/pkgconfig
        )
}

# Install SPOUT
wget --no-verbose https://frakira.fi.muni.cz/~xpulec/SpoutSDK.zip # this is the SDK subdirectory installed by Spout installer
unzip SpoutSDK.zip -d src
MSBuild.exe -p:PlatformToolset=v142  -p:Configuration=Release -p:Platform=x64 src/SpoutSDK/VS2012
data/scripts/build_spout64.sh src/SpoutSDK/VS2012/x64/Release

# Install FFMPEG
wget --no-verbose https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z && 7z x ffmpeg-release-full-shared.7z && cp -r ffmpeg-*build-shared/{bin,lib,include} /usr/local && rm -rf ffmpeg-* || exit 1

# Install GPUJPEG
( wget --no-verbose https://github.com/CESNET/GPUJPEG/releases/download/continuous/GPUJPEG.zip && unzip GPUJPEG.zip && cp -r GPUJPEG/* /usr/local )

build_cineform
