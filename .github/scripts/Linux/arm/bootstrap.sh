#!/bin/sh -eu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

ARCH=$1
OLDPWD=$(pwd)

if grep -q Raspbian /etc/os-release; then # https://bugs.launchpad.net/ubuntu/+source/qemu/+bug/1670905 workaround
        sed -i s-http://deb.debian.org/debian-http://mirrordirector.raspbian.org/raspbian/- /etc/apt/sources.list
        apt -y install curl
        curl http://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -
        echo 'deb http://archive.raspberrypi.org/debian buster main' >> /etc/apt/sources.list
        apt -y update
fi

apt -y install build-essential git pkg-config autoconf automake libtool
apt -y install portaudio19-dev libsdl2-dev libglib2.0-dev libglew-dev libcurl4-openssl-dev freeglut3-dev libssl-dev libjack-dev libasound2-dev

# FFmpeg
if [ $ARCH = armhf ]; then # Raspbian - build own FFmpeg with OMX camera patch
        git clone --depth 1 https://github.com/raspberrypi/firmware.git firmware && mv firmware/* / && echo /opt/vc/lib > /etc/ld.so.conf.d/00-vmcs.conf && ldconfig
        sed -i '/^deb /p;s/deb/deb-src/' /etc/apt/sources.list
        apt -y update && apt -y build-dep ffmpeg
        apt -y remove libavcodec58 && apt -y autoremove
        git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg
        git fetch --depth 2 https://github.com/Serveurperso/FFmpeg.git && git cherry-pick FETCH_HEAD
        ./configure --enable-gpl --disable-stripping --enable-libaom --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libspeex --enable-libvpx --enable-libwebp --enable-libx265 --enable-omx --enable-neon --enable-libx264 --enable-mmal --enable-omx-rpi --cpu=arm1176jzf-s --enable-shared --disable-static
        make -j 3 install
        cd $OLDPWD
else
        apt -y install libavcodec-dev libavformat-dev libswscale-dev
fi

# appimagetool
apt -y install desktop-file-utils git-core libfuse-dev libcairo2-dev cmake wget zsync
git clone -b 12 https://github.com/AppImage/AppImageKit.git
cd AppImageKit && patch -N -p1 < /mksquashfs-compilation-fix.patch
./build.sh
cd build
cmake -DAUXILIARY_FILES_DESTINATION= ..
make -j 3 install
cd $OLDPWD

rm -rf FFmpeg AppImageKit
