#!/bin/sh -exu

debver=$(. /etc/os-release; echo "$VERSION_CODENAME")
readonly debver

apt -y install curl gnupg
echo -k > ~/.curlrc

if grep -q Debian /etc/os-release; then
        cat >/etc/apt/sources.list <<EOF
deb http://deb.debian.org/debian $debver main contrib non-free
deb http://security.debian.org/debian-security $debver-security main contrib non-free
deb http://deb.debian.org/debian $debver-updates main contrib non-free
EOF
fi
curl http://archive.raspberrypi.org/debian/raspberrypi.gpg.key -o \
        /usr/share/keyrings/raspberrypi.gpg.key
cat >>/etc/apt/sources.list <<EOF
deb [signed-by=/usr/share/keyrings/raspberrypi.gpg.key] \
http://archive.raspberrypi.org/debian $debver main
EOF
apt -y update

if [ "$debver" = buster ]; then
        raspbian_build_sdl2() { (
                sed -i '/^deb /p;s/deb /deb-src /' /etc/apt/sources.list
                apt -y update
                apt-get -y build-dep libsdl2-dev
                apt -y install libgbm-dev
                readonly sdl_ver=2.0.22
                curl -k -LO https://www.libsdl.org/release/SDL2-$sdl_ver.tar.gz
                tar xaf SDL2-$sdl_ver.tar.gz
                cd SDL2-$sdl_ver
                ./configure --enable-video-kmsdrm
                make -j "$(nproc)" install
        ); }
        # 3.16 in the added repository is broken with chrooted qemu-user-static
        apt -y install cmake=3.13.4-1 cmake-data=3.13.4-1
        raspbian_build_sdl2
else
        apt -y install cmake libsdl2-dev
fi

apt -y install autoconf automake build-essential git pkg-config libtool sudo
apt -y install libcurl4-openssl-dev libsoxr-dev libspeexdsp-dev libssl-dev
apt -y install libasound2-dev portaudio19-dev libjack-dev
apt -y install libglew-dev libglfw3-dev libglm-dev
apt -y install libcaca-dev libmagickwand-dev libnatpmp-dev libopencv-core-dev libopencv-imgproc-dev libv4l-dev
apt -y install libavcodec-dev libavformat-dev libswscale-dev libraspberrypi-dev

/.github/scripts/install-common-deps.sh
/.github/scripts/Linux/install_others.sh ndi pipewire ximea

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
