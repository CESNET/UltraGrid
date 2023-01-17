#!/bin/sh -eu

mkdir -p /var/tmp/sdl
cd /var/tmp/sdl

# v2.0.22 requires wayland-client version 1.18.0 but in U18.04 is only 1.16.0
curl -sSLO https://github.com/libsdl-org/SDL/releases/download/release-2.0.20/SDL2-2.0.20.tar.gz
tar xaf SDL2*tar*
cd SDL2-2.0.20
./configure
make -j "$(nproc)"
sudo make install # SDL needs to be installed here because it is a dependency for the below
cd ..

git clone -b SDL2 --depth 1 https://github.com/libsdl-org/SDL_mixer
cd SDL_mixer
./configure
make -j "$(nproc)"
sudo make install
cd ..

# v2.0.18 and further require automake 1.16 but U18.04 has only automake 1.15.1
git clone -b release-2.0.15 --depth 1 https://github.com/libsdl-org/SDL_ttf
cd SDL_ttf
./autogen.sh # to allow automake 1.15.1
./configure
make -j "$(nproc)"
sudo make install
cd ..

