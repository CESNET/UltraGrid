#!/bin/sh -eu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

ARCH=$1
BUILD_DIR=$2

sudo chroot $BUILD_DIR /bin/sh -c 'if grep -q Raspbian /etc/os-release; then sed -i s-http://deb.debian.org/debian-http://mirrordirector.raspbian.org/raspbian/- /etc/apt/sources.list && apt-get -y update; fi' # https://bugs.launchpad.net/ubuntu/+source/qemu/+bug/1670905 workaround
sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install build-essential git pkg-config autoconf automake libtool'
sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install portaudio19-dev libsdl2-dev libglib2.0-dev libglew-dev libcurl4-openssl-dev freeglut3-dev libssl-dev libjack-dev libasound2-dev'
if [ $ARCH = armhf ]; then # Raspbian - build own FFmpeg with OMX camera patch
        sudo chroot $BUILD_DIR /bin/sh -c "git clone --depth 1 https://github.com/raspberrypi/firmware.git firmware && mv firmware/* / && echo /opt/vc/lib > /etc/ld.so.conf.d/00-vmcs.conf && ldconfig"\
"&& sed -i '/^deb /p;s/deb/deb-src/' /etc/apt/sources.list "\
"&& apt-get -y update && apt-get -y build-dep ffmpeg"\
"&& apt-get -y remove libavcodec58 && apt-get -y autoremove"\
"&& git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg"\
"&& git fetch --depth 2 https://github.com/Serveurperso/FFmpeg.git && git cherry-pick FETCH_HEAD"\
"&& ./configure --enable-gpl --disable-stripping --enable-libaom --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libspeex --enable-libvpx --enable-libwebp --enable-libx265 --enable-omx --enable-neon --enable-libx264 --enable-mmal --enable-omx-rpi --cpu=arm1176jzf-s --enable-shared --disable-static && make -j 3 && make install"\
"&& cd .. && rm -rf FFmpeg"
else
        sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install libavcodec-dev libavformat-dev libswscale-dev'
fi
sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install desktop-file-utils git-core libfuse-dev libcairo2-dev cmake wget zsync' # to build appimagetool
sudo chroot $BUILD_DIR /bin/sh -c 'git clone https://github.com/AppImage/AppImageKit.git && cd AppImageKit && ./build.sh && cd build && cmake -DAUXILIARY_FILES_DESTINATION= .. && make install'
sudo chroot $BUILD_DIR /bin/sh -c 'rm -rf AppImageKit; apt-get -y clean'
