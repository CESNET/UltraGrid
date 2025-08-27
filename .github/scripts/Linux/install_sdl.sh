#!/bin/sh -eu

cd /var/tmp/sdl
sudo cmake --install SDL/build
sudo cmake --install SDL_ttf/build
sudo cmake --install fluidsynth/build

