#!/bin/sh -exu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

ARCH=$1
OLDPWD=$(pwd)
CMAKE_FORCE_VER=

raspbian_build_sdl2() {
        (
        apt-get -y build-dep libsdl2-dev
        apt -y install libgbm-dev
        SDL_VER=2.0.10 # 2.0.14 doesn't compile with Rasbpian 10
        curl -k -LO https://www.libsdl.org/release/SDL2-$SDL_VER.tar.gz
        tar xaf SDL2-$SDL_VER.tar.gz
        cd SDL2-$SDL_VER
        ./configure --enable-video-kmsdrm
        make -j $(nproc) install
        )
}

if grep -q Raspbian /etc/os-release; then # https://bugs.launchpad.net/ubuntu/+source/qemu/+bug/1670905 workaround
        sed -i s-http://deb.debian.org/debian-http://mirrordirector.raspbian.org/raspbian/- /etc/apt/sources.list
        apt -y install curl
        curl http://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -
        echo 'deb http://archive.raspberrypi.org/debian buster main' >> /etc/apt/sources.list
        apt -y update
        CMAKE_FORCE_VER="=3.13.4-1" # solves https://gitlab.kitware.com/cmake/cmake/-/issues/20568
fi

apt -y install build-essential git pkg-config autoconf automake libtool
apt -y install portaudio19-dev libglib2.0-dev libglew-dev libcurl4-openssl-dev freeglut3-dev libssl-dev libjack-dev libasound2-dev

# FFmpeg
if [ $ARCH = armhf ]; then # Raspbian - build own FFmpeg with OMX camera patch
        apt -y install libraspberrypi-dev libdrm-dev
        sed -i '/^deb /p;s/deb /deb-src /' /etc/apt/sources.list
        apt -y update && apt -y build-dep ffmpeg
        raspbian_build_sdl2
        apt -y remove libavcodec58 && apt -y autoremove
        git clone https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg
        git checkout n4.3.3

        # apply patches
        FF_PATCH_DIR=/ffmpeg-arm-patches
        for n in `ls $FF_PATCH_DIR`; do
                git apply $FF_PATCH_DIR/$n
        done


        ./configure --enable-gpl --disable-stripping --enable-libaom --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libspeex --enable-libvpx --enable-libwebp --enable-libx265 --enable-omx --enable-neon --enable-libx264 --enable-mmal --enable-omx-rpi --enable-rpi --enable-vout-drm --enable-libdrm --enable-v4l2-request --enable-libudev --cpu=arm1176jzf-s --enable-shared --disable-static
        make -j3 install
        cd $OLDPWD
else
        apt -y install libavcodec-dev libavformat-dev libsdl2-dev libswscale-dev
fi

# appimagetool
apt -y install desktop-file-utils git-core libfuse-dev libcairo2-dev cmake$CMAKE_FORCE_VER cmake-data$CMAKE_FORCE_VER wget zsync
git clone -b 12 https://github.com/AppImage/AppImageKit.git
cd AppImageKit && patch -N -p1 < /mksquashfs-compilation-fix.patch
./build.sh
cd build
cmake -DAUXILIARY_FILES_DESTINATION= ..
make -j 3 install
cd $OLDPWD

rm -rf FFmpeg AppImageKit
