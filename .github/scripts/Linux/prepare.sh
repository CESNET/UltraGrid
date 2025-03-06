#!/bin/bash -eux

export PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig:/usr/local/lib/pkgconfig
printf "%b" "\
CPATH=/usr/local/qt/include\n\
LIBRARY_PATH=/usr/local/qt/lib\n\
PKG_CONFIG_PATH=$PKG_CONFIG_PATH\n" >> "$GITHUB_ENV"
printf "/usr/local/qt/bin\n" >> "$GITHUB_PATH"

git config --global user.name "UltraGrid Builder"
git config --global user.email "ultragrid@example.org"

# add deb-src for build-dep ffmpeg
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then # deb822 (new) format
        sudo sed -i 's/Types: deb/Types: deb deb-src/' \
                /etc/apt/sources.list.d/ubuntu.sources
else # one-line-style (old) format
        sed -n '/^deb /s/^deb /deb-src /p' /etc/apt/sources.list |
                sudo tee /etc/apt/sources.list.d/sources.list
fi
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
sudo apt install libcurl4-openssl-dev # for RTSP client (vidcap)
sudo apt install i965-va-driver-shaders libva-dev # instead of i965-va-driver
sudo apt-mark hold libva2
sudo apt install uuid-dev # Cineform

(
        . /etc/os-release
        if [ "$ID" != ubuntu ] || [ "$VERSION_ID" != 20.04 ]; then
                exit
        fi
        sudo apt install gcc-10 g++-10
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10

        # FFmpeg deps
        for i in $(seq 10); do
                [ "$i" -gt 1 ] && sleep $((2**i))
                # openh264, new x265
                sudo add-apt-repository --yes ppa:savoury1/ffmpeg4 && err=0 && break || err=$?
        done
        (exit "$err")
)

get_build_deps_excl() { # $2 - pattern to exclude; separate packates with '\|' (BRE alternation)
        apt-cache showsrc "$1" | sed -n '/^Build-Depends:/{s/Build-Depends://;p;q}' | tr ',' '\n' | cut -f 2 -d\  | grep -v "$2"
}
sudo apt build-dep libsdl2
sdl2_mix_build_dep=$(get_build_deps_excl libsdl2-mixer libsdl2-dev)
sdl2_ttf_build_dep=$(get_build_deps_excl libsdl2-ttf libsdl2-dev)
# shellcheck disable=SC2086 # intentional
sudo apt install $sdl2_mix_build_dep $sdl2_ttf_build_dep

# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
ffmpeg_build_dep=$(get_build_deps_excl ffmpeg 'libva-dev')
# shellcheck disable=SC2086 # intentional
sudo apt install $ffmpeg_build_dep libdav1d-dev libde265-dev \
        libopenh264-dev libvulkan-dev
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' libvpx-dev nginx

sudo apt install qtbase5-dev

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

"$GITHUB_WORKSPACE/.github/scripts/Linux/install_others.sh"

