#!/bin/sh -eu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

BUILD_DIR=$2

sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install build-essential pkg-config autoconf automake libtool'
sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install portaudio19-dev libsdl2-dev libglib2.0-dev libglew-dev libcurl4-openssl-dev freeglut3-dev libssl-dev libjack-dev libavcodec-dev libswscale-dev libasound2-dev'
sudo chroot $BUILD_DIR /bin/sh -c 'apt-get -y install desktop-file-utils git-core libfuse-dev libcairo2-dev cmake wget zsync' # to build appimagetool
sudo chroot $BUILD_DIR /bin/sh -c 'git clone https://github.com/AppImage/AppImageKit.git && cd AppImageKit && ./build.sh && cd build && cmake -DAUXILIARY_FILES_DESTINATION= .. && make install'
sudo chroot $BUILD_DIR /bin/sh -c 'rm -rf AppImageKit; apt-get -y clean'
