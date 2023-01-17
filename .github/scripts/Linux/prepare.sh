#!/bin/bash -eux

# shellcheck disable=SC2140
printf "%b" "AJA_DIRECTORY=/var/tmp/ntv2\n"\
"CPATH=/usr/local/qt/include\n"\
"LIBRARY_PATH=/usr/local/qt/lib\n"\
"PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig\n" >> "$GITHUB_ENV"
printf "/usr/local/qt/bin\n" >> "$GITHUB_PATH"

git config --global user.name "UltraGrid Builder"
git config --global user.email "ultragrid@example.org"

sed -n '/^deb /s/^deb /deb-src /p' /etc/apt/sources.list | sudo tee /etc/apt/sources.list.d/sources.list # for build-dep ffmpeg
sudo apt update
sudo apt install appstream # appstreamcli for mkappimage AppStream validation
sudo apt install fonts-dejavu-core
sudo apt install libcppunit-dev
sudo apt --no-install-recommends install nvidia-cuda-toolkit
sudo apt install libglew-dev libglfw3-dev
sudo apt install libglm-dev
sudo apt install libx11-dev
sudo apt install libsoxr-dev libspeexdsp-dev
sudo apt install libssl-dev
sudo apt install libasound-dev libjack-jackd2-dev libnatpmp-dev libv4l-dev portaudio19-dev
sudo apt install libopencv-core-dev libopencv-imgproc-dev
sudo apt install libcurl4-nss-dev
sudo apt install i965-va-driver-shaders # instead of i965-va-driver
sudo apt install uuid-dev # Cineform

get_build_deps_excl() { # $2 - pattern to exclude
        apt-cache showsrc "$1" | sed -n '/^Build-Depends:/{s/Build-Depends://;p;q}' | tr ',' '\n' | cut -f 2 -d\  | grep -v "$2"
}
sudo apt build-dep libsdl2
sdl2_mix_build_dep=$(get_build_deps_excl libsdl2-mixer libsdl2-dev)
sdl2_ttf_build_dep=$(get_build_deps_excl libsdl2-ttf libsdl2-dev)
# shellcheck disable=SC2086 # intentional
sudo apt install $sdl2_mix_build_dep $sdl2_ttf_build_dep

# FFmpeg deps
sudo add-apt-repository ppa:savoury1/vlc3 # new x265
# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
ffmpeg_build_dep=$(get_build_deps_excl ffmpeg 'libzmq3-dev\|libsdl2-dev')
# shellcheck disable=SC2086 # intentional
sudo apt install $ffmpeg_build_dep libdav1d-dev libde265-dev
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' libvpx-dev 'libx264*' nginx
# own x264 build
sudo apt --no-install-recommends install asciidoc xmlto
# openVPL
sudo apt install libva-dev libdrm-dev libx11-dev libx11-xcb-dev libxcb-present-dev libxcb-dri3-dev
sudo curl -LO http://azure.archive.ubuntu.com/ubuntu/pool/main/w/wayland-protocols/wayland-protocols_1.20-1_all.deb # at least 1.15 is needed
sudo dpkg -i wayland-protocols_*_all.deb

sudo apt install qtbase5-dev

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

"$GITHUB_WORKSPACE/.github/scripts/Linux/install_others.sh"

