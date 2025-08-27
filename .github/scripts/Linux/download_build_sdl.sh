#!/bin/sh -eu

mkdir -p /var/tmp/sdl
cd /var/tmp/sdl

git clone --depth 1 https://github.com/libsdl-org/SDL
cd SDL
cmake -S . -B build
cmake --build build -j "$(nproc)"
sudo cmake --install build
cd ..

git clone --depth 1 https://github.com/libsdl-org/SDL_ttf
cd SDL_ttf
cmake -S . -B build
cmake --build build -j "$(nproc)"
sudo cmake --install build
cd ..

