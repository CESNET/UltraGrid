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

apt -y install curl
echo -k > ~/.curlrc

if grep -q Raspbian /etc/os-release; then # https://bugs.launchpad.net/ubuntu/+source/qemu/+bug/1670905 workaround
        sed -i s-http://deb.debian.org/debian-http://mirrordirector.raspbian.org/raspbian/- /etc/apt/sources.list
        curl http://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -
        echo 'deb http://archive.raspberrypi.org/debian buster main' >> /etc/apt/sources.list
        apt -y update
fi

apt -y install build-essential git pkg-config autoconf automake libtool
apt -y install libcurl4-openssl-dev libsoxr-dev libssl-dev
apt -y install libasound2-dev portaudio19-dev libjack-dev
apt -y install libglew-dev libglfw3-dev libglm-dev

# FFmpeg
if [ "$ARCH" = armhf ]; then # Raspbian - build own FFmpeg with OMX camera patch
        apt -y install libraspberrypi-dev libdrm-dev
        sed -i '/^deb /p;s/deb /deb-src /' /etc/apt/sources.list
        apt -y update && apt -y build-dep ffmpeg
        raspbian_build_sdl2
        apt -y remove libavcodec58 && apt -y autoremove
        git clone --depth 1 -b n4.3.3 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg

        # apply patches
        find /ffmpeg-arm-patches -name '*.patch' -print0 | sort -z | xargs -0 -n 1 git apply

        ./configure --enable-gpl --disable-stripping --enable-libaom --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libspeex --enable-libvpx --enable-libwebp --enable-libx265 --enable-omx --enable-neon --enable-libx264 --enable-mmal --enable-omx-rpi --enable-rpi --enable-vout-drm --enable-libdrm --enable-v4l2-request --enable-libudev --cpu=arm1176jzf-s --enable-shared --disable-static
        make -j3 install
        cd "$OLDPWD"
else
        apt -y install libavcodec-dev libavformat-dev libsdl2-dev libswscale-dev
fi

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
