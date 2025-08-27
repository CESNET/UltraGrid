#!/bin/sh -eu

cd /var/tmp/sdl/SDL
sudo cmake --install build
cd ../SDL_ttf
sudo cmake --install build

