#!/bin/sh -exu

# If changing the file, do not forget to regenerate cache in ARM Build GitHub action

OLDPWD=$(pwd)
readonly debver=bullseye

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

apt -y install autoconf automake build-essential cmake git pkg-config libtool sudo
apt -y install libcurl4-openssl-dev libsoxr-dev libspeexdsp-dev libssl-dev
apt -y install libasound2-dev portaudio19-dev libjack-dev
apt -y install libglew-dev libglfw3-dev libglm-dev
apt -y install libcaca-dev libmagickwand-dev libnatpmp-dev libopencv-core-dev libopencv-imgproc-dev
apt -y install libavcodec-dev libavformat-dev libsdl2-dev libswscale-dev libraspberrypi-dev

/.github/scripts/install-common-deps.sh
/.github/scripts/Linux/install_others.sh ndi
/.github/scripts/Linux/install_others.sh ximea

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
