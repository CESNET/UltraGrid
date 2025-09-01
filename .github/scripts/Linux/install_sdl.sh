#!/bin/sh -eu

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/common.sh" # for get_build_deps_excl

# build dir that will be restored from cache
cache_dir=/var/tmp/sdl
features="-DSDL_KMSDRM=ON\
 -DSDL_OPENGL=ON\
 -DSDL_VULKAN=ON\
 -DSDL_WAYLAND=ON\
 -DSDL_X11=ON"

# install the deps - runs always (regardless the cache)
deps() {
        sudo apt build-dep libsdl2
        fluidsynth_build_dep=$(get_build_deps_excl libfluidsynth3 libsdl2-dev)
        sdl2_ttf_build_dep=$(get_build_deps_excl libsdl2-ttf libsdl2-dev)
        # shellcheck disable=SC2086 # intentional
        sudo apt install $fluidsynth_build_dep $sdl2_ttf_build_dep
}

# build SDL, SDL_ttf and fluidsynth and also install them
build_install() {
        mkdir -p $cache_dir
        cd $cache_dir

        git clone --depth 1 https://github.com/libsdl-org/SDL
        cmake -S SDL -B SDL/build
        cmake --build SDL/build -j "$(nproc)"
        sudo cmake --install SDL/build

        git clone --depth 1 https://github.com/libsdl-org/SDL_ttf
        cmake -S SDL_ttf -B SDL_ttf/build
        cmake --build SDL_ttf/build -j "$(nproc)"
        sudo cmake --install SDL_ttf/build

        git clone --recurse-submodules --depth 1\
         https://github.com/Fluidsynth/fluidsynth
        # shellcheck disable=SC2086 # intentional
        cmake $features -S fluidsynth -B fluidsynth/build
        cmake --build fluidsynth/build -j "$(nproc)"
        sudo cmake --install fluidsynth/build
}

# if cache is successfully restored, just install the builds
install_cached() {
        cd $cache_dir
        sudo cmake --install SDL/build
        sudo cmake --install SDL_ttf/build
        sudo cmake --install fluidsynth/build
}

deps
if [ -d $cache_dir ]; then
        install_cached
else
        build_install
fi
