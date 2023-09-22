#!/bin/sh -exu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

ARCH=$1
OLDPWD=$(pwd)

raspbian_build_sdl2() {
        (
        apt-get -y build-dep libsdl2-dev
        apt -y install libgbm-dev
        SDL_VER=2.0.10 # 2.0.14 doesn't compile with Rasbpian 10
        curl -k -LO https://www.libsdl.org/release/SDL2-$SDL_VER.tar.gz
        tar xaf SDL2-$SDL_VER.tar.gz
        cd SDL2-$SDL_VER
        ./configure --enable-video-kmsdrm
        make -j "$(nproc)" install
        )
}

apt -y install curl gnupg
echo -k > ~/.curlrc

if grep -q Raspbian /etc/os-release; then
        curl http://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -
        echo 'deb http://archive.raspberrypi.org/debian buster main' >> /etc/apt/sources.list
        apt -y update
        apt -y install libraspberrypi-dev
fi

apt -y install cmake=3.13.4-1 cmake-data=3.13.4-1 # 3.16 in the above added repository is broken with chrooted qemu-user-static

apt -y install autoconf automake build-essential git pkg-config libtool sudo
apt -y install libcurl4-openssl-dev libsoxr-dev libspeexdsp-dev libssl-dev
apt -y install libasound2-dev portaudio19-dev libjack-dev
apt -y install libglew-dev libglfw3-dev libglm-dev
apt -y install libcaca-dev libmagickwand-dev libnatpmp-dev libopencv-core-dev libopencv-imgproc-dev

/.github/scripts/install-common-deps.sh
/.github/scripts/Linux/install_others.sh ndi
/.github/scripts/Linux/install_others.sh ximea

# FFmpeg
apt -y install libavcodec-dev libavformat-dev libsdl2-dev libswscale-dev

# mkappimage
mkai_arch=$(dpkg --print-architecture)
if [ "$mkai_arch" = arm64 ]; then
        mkai_arch=aarch64
fi
mkai_url=$(curl https://api.github.com/repos/probonopd/go-appimage/releases/tags/continuous | grep "browser_download_url.*mkappimage-.*-$mkai_arch.AppImage" | head -n 1 | cut -d '"' -f 4)
curl -L "$mkai_url" > mkappimage
chmod 755 mkappimage
#shellcheck disable=SC2211
/usr/bin/qemu-*-static ./mkappimage --appimage-extract
mv squashfs-root /opt/mkappimage
printf '%b' '#!/bin/sh\nexec /opt/mkappimage/AppRun "$@"\n' > /usr/local/bin/mkappimage
chmod 755 /usr/local/bin/mkappimage

rm -rf FFmpeg AppImageKit mkappimage
