#!/bin/sh -eux

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export MSYSTEM_PREFIX=/clang64
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

PACMAN_INSTALL='pacman -Sy --needed --noconfirm --disable-download-timeout'
# Install MSYS2 packages
MINGW_PACKAGE_PREFIX=mingw-w64-clang-x86_64
$PACMAN_INSTALL automake autoconf git make pkgconf ${MINGW_PACKAGE_PREFIX}-toolchain ${MINGW_PACKAGE_PREFIX}-cppunit unzip zip
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-ffmpeg
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-speexdsp
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-glew ${MINGW_PACKAGE_PREFIX}-SDL2 ${MINGW_PACKAGE_PREFIX}-SDL2_mixer ${MINGW_PACKAGE_PREFIX}-SDL2_ttf ${MINGW_PACKAGE_PREFIX}-glfw
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-glm
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-curl # RTSP capture
pacman -Scc --noconfirm # make some free space
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-qt6-base ${MINGW_PACKAGE_PREFIX}-qt6-tools
$PACMAN_INSTALL ${MINGW_PACKAGE_PREFIX}-imagemagick ${MINGW_PACKAGE_PREFIX}-opencv
$PACMAN_INSTALL p7zip
$PACMAN_INSTALL libtool # PCP
pacman -Scc --noconfirm

# Build AJA wrapper if we have SDK
install_aja() {(
        git clone --depth 1 https://github.com/aja-video/ntv2 AJA
        cd AJA
        AJA_GH_PATH=https://github.com/$(curl https://github.com/aja-video/ntv2/releases  | grep libs_windows_ | head -n 1 | cut -d '"' -f 2)
        curl -L "$AJA_GH_PATH" -o aja_build.zip
        rm README.md # would be overriden from zip below
        unzip aja_build.zip
        mkdir -p lib
        cp Release/*.lib lib
        cp Release/*.dll /usr/local/bin
        cd ..
        data/scripts/build_aja_lib_win64.sh
)}

# DELTACAST
if [ -n "$SDK_URL" ]; then
        mkdir VideoMaster
        cd VideoMaster
        if curl -f -S "$SDK_URL/VideoMaster_SDK_Windows.zip" -O; then
                FEATURES="$FEATURES --enable-deltacast"
                echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
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
        cd "$GITHUB_WORKSPACE/cineform-sdk/build"
        sed -i 's/GetProcessorCount/_&/'  ../ConvertLib/ImageScaler.cpp # workaround for needless Win GetProcessorCount definition
        cmake -DBUILD_STATIC=false -DBUILD_TOOLS=false -A x64 .. # assume "-G 'Visual Studio 16 2019'"
        cmake --build . --config Release --parallel
        cp Release/CFHDCodec.dll /usr/local/bin && cp Release/CFHDCodec.lib /usr/local/lib && cp ../Common/* /usr/local/include && cp libcineformsdk.pc /usr/local/lib/pkgconfig
        )
}

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

"$GITHUB_WORKSPACE/.github/scripts/Windows/install_natpmp.sh"
"$GITHUB_WORKSPACE/.github/scripts/Windows/install_spout.sh"

# Install GPUJPEG
( wget --no-verbose https://github.com/CESNET/GPUJPEG/releases/download/continuous/GPUJPEG.zip && unzip GPUJPEG.zip && cp -r GPUJPEG/* /usr/local || exit 1 )

build_cineform
install_aja

