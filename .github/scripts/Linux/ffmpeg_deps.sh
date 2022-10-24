#!/bin/sh -eux
#
## Installs common FFmpeg build and runtime (for FFmpeg cache restore) dependencies.

sudo add-apt-repository ppa:savoury1/vlc3 # new x265

# updates nasm 2.13->2.14 in U18.04 (needed for rav1e)
update_nasm() {
        if [ -z "$(apt-cache search --names-only '^nasm-mozilla$')" ]; then
                return
        fi
        sudo apt install nasm- nasm-mozilla
        sudo ln -s /usr/lib/nasm-mozilla/bin/nasm /usr/bin/nasm
}

# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
sudo aptitude -y build-dep ffmpeg libsdl2-dev: libzmq3-dev:
sudo apt install libdav1d-dev
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' libvpx-dev 'libx264*' nginx
update_nasm
# own x264 build
sudo apt --no-install-recommends install asciidoc xmlto

