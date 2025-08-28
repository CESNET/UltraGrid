#!/bin/bash -eux

dir=$(dirname "$0")

export PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig:/usr/local/lib/pkgconfig
export LIBRARY_PATH=/usr/local/lib:/usr/local/qt/lib
printf "%b" "\
CPATH=/usr/local/qt/include\n\
LIBRARY_PATH=$LIBRARY_PATH\n\
LD_LIBRARY_PATH=$LIBRARY_PATH\n\
PKG_CONFIG_PATH=$PKG_CONFIG_PATH\n" >> "$GITHUB_ENV"
printf "/usr/local/qt/bin\n" >> "$GITHUB_PATH"

# add deb-src for build-dep ffmpeg
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then # deb822 (new) format
        sudo sed -i 's/Types: deb/Types: deb deb-src/' \
                /etc/apt/sources.list.d/ubuntu.sources
else # one-line-style (old) format
        sed -n '/^deb /s/^deb /deb-src /p' /etc/apt/sources.list |
                sudo tee /etc/apt/sources.list.d/sources.list
fi
sudo apt-mark hold libsdl2-2.0-0
sudo apt update
sudo apt install appstream `# appstreamcli for mkappimage AppStream validation` \
        asciidoc
sudo apt install fonts-dejavu-core
sudo apt --no-install-recommends install nvidia-cuda-toolkit
sudo apt install libglew-dev
sudo apt install libglm-dev
sudo apt install imagemagick libmagickwand-dev
sudo apt install libsoxr-dev libspeexdsp-dev
sudo apt install libssl-dev
sudo apt install libasound-dev libcaca-dev libjack-jackd2-dev libnatpmp-dev libv4l-dev portaudio19-dev
sudo apt install libopencv-core-dev libopencv-imgproc-dev
sudo apt install libcurl4-openssl-dev # for RTSP client (vidcap)
sudo apt install i965-va-driver-shaders libva-dev # instead of i965-va-driver

sudo apt install qt6-base-dev qt6-wayland
. /etc/os-release # source ID and VERSION_ID
if [ "$ID" = ubuntu ] && [ "$VERSION_ID" = 22.04 ]; then
        # https://bugs.launchpad.net/ubuntu/+source/qtchooser/+bug/1964763 bug
        # workaround proposed in https://askubuntu.com/a/1460243
        sudo qtchooser -install qt6 "$(command -v qmake6)"
        sudo ln -n "/usr/lib/$(uname -m)-linux-gnu/qt-default/qtchooser/\
qt6.conf" "/usr/lib/$(uname -m)-linux-gnu/qt-default/qtchooser/default.conf"
 fi

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

"$GITHUB_WORKSPACE/.github/scripts/Linux/install_others.sh"

"$dir"/install_sdl.sh
"$dir"/install_ffmpeg.sh
"$dir"/install_glfw.sh

