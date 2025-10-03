#!/bin/bash -eux

export PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig:/usr/local/lib/pkgconfig
printf "%b" "AJA_DIRECTORY=/var/tmp/ntv2\n\
CPATH=/usr/local/qt/include\n\
LIBRARY_PATH=/usr/local/qt/lib\n\
PKG_CONFIG_PATH=$PKG_CONFIG_PATH\n" >> "$GITHUB_ENV"
printf "/usr/local/qt/bin\n" >> "$GITHUB_PATH"

sed -n '/^deb /s/^deb /deb-src /p' /etc/apt/sources.list | sudo tee /etc/apt/sources.list.d/sources.list # for build-dep ffmpeg
sudo apt update
sudo apt install appstream `# appstreamcli for mkappimage AppStream validation` \
        asciidoc
sudo apt install fonts-dejavu-core
sudo apt --no-install-recommends install nvidia-cuda-toolkit
sudo apt install libglew-dev libglfw3-dev
sudo apt install libglm-dev
sudo apt install libmagickwand-dev
sudo apt install libsdl2-dev libsdl2-mixer-dev libsdl2-ttf-dev
sudo apt install libsoxr-dev libspeexdsp-dev
sudo apt install libssl-dev
sudo apt install libasound-dev libcaca-dev libjack-jackd2-dev libnatpmp-dev libv4l-dev portaudio19-dev
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

# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
ffmpeg_build_dep=$(get_build_deps_excl ffmpeg 'libzmq3-dev\|libsdl2-dev')
# shellcheck disable=SC2086 # intentional
sudo apt install $ffmpeg_build_dep libde265-dev meson
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' libvpx-dev 'libx264*' nginx
# own x264 build
sudo apt --no-install-recommends install asciidoc xmlto

sudo apt install qtbase5-dev

. "$GITHUB_WORKSPACE/.github/scripts/defs.sh"
sf=$(basename "$DEFAULT_SF_URL")
curl -L "$DEFAULT_SF_URL" -o "$HOME/$sf"
printf '%b' "SDL_SOUNDFONTS=$HOME/$sf\n" >> "$GITHUB_ENV"

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

"$GITHUB_WORKSPACE/.github/scripts/Linux/install_others.sh"

