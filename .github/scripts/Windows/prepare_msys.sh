#!/bin/bash

set -ex

mkdir -p /usr/local/lib /usr/local/bin /usr/local/include
cat >> ~/.bash_profile <<'EOF'
export PATH=/mingw64/bin:/usr/local/bin:$PATH
export CPATH=/usr/local/include
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/mingw64/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib
EOF

CUDA_D=$(ls -d /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/*)
if test -d "$CUDA_D"; then
        echo "export CPATH=\$CPATH:'$CUDA_D/include'" >> ~/.bash_profile
fi

if test -d /c/Program\ Files/NewTek; then
        NDI_D=$(ls -d /c/Program\ Files/NewTek/*SDK)
        echo "export CPATH=\$CPATH:'$NDI_D/Include'" >> ~/.bash_profile
        echo "export LIBRARY_PATH=\$LIBRARY_PATH:'$NDI_D/Lib/x64'" >> ~/.bash_profile
fi

echo cd `cygpath $GITHUB_WORKSPACE` >> ~/.bash_profile

. ~/.bash_profile

# Install MSYS2 packages
pacman -Sy --noconfirm --disable-download-timeout automake autoconf git make pkg-config mingw-w64-x86_64-toolchain mingw-w64-x86_64-cppunit unzip zip
pacman -Sy --noconfirm --disable-download-timeout mingw-w64-x86_64-glew mingw-w64-x86_64-SDL2 mingw-w64-x86_64-freeglut
pacman -Sy --noconfirm --disable-download-timeout mingw-w64-x86_64-portaudio # in case of problems build PA with --with-winapi=wmme,directx,wasapi
pacman -Scc --noconfirm # make some free space
pacman -Sy --noconfirm --disable-download-timeout mingw-w64-x86_64-qt5
pacman -Scc --noconfirm

# Build AJA wrapper if we have SDK
if test -d /c/AJA; then
        data/scripts/build_aja_lib_win64.sh
fi

# Install FFMPEG
wget --no-verbose https://ffmpeg.zeranoe.com/builds/win64/dev/ffmpeg-latest-win64-dev.zip && wget --no-verbose https://ffmpeg.zeranoe.com/builds/win64/shared/ffmpeg-latest-win64-shared.zip && unzip ffmpeg-latest-win64-dev.zip && unzip ffmpeg-latest-win64-shared.zip && cp -r ffmpeg-latest-win64-dev/include/* /usr/local/include && cp -r ffmpeg-latest-win64-dev/lib/* /usr/local/lib && cp -r ffmpeg-latest-win64-shared/bin/* /usr/local/bin && rm -rf ffmpeg-latest-*

# Build GPUJPEG
( cd gpujpeg && nvcc -I. -DGPUJPEG_EXPORTS -o gpujpeg.dll --shared src/gpujpeg_*c src/gpujpeg*cu && cp gpujpeg.lib /usr/local/lib && cp gpujpeg.dll /usr/local/bin && cp -r libgpujpeg /usr/local/include )

# Build CineForm
( cd cineform-sdk && cmake -DBUILD_STATIC=false -G Visual\ Studio\ 16\ 2019 -A x64 && MSBuild.exe CineFormSDK.sln -property:Configuration=Release && cp Release/CFHDCodec.dll /usr/local/bin && cp Release/CFHDCodec.lib /usr/local/lib && cp Common/* /usr/local/include && cp libcineformsdk.pc /usr/local/lib/pkgconfig )

