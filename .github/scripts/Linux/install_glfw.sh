#!/bin/sh -eu

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/common.sh" # for get_build_deps_excl

# build dir that will be restored from cache
cache_dir=/var/tmp/glfw

# install the deps - runs always (regardless the cache)
deps() {
        sudo apt build-dep libglfw3
}

# build SDL, SDL_ttf and fluidsynth and also install them
build_install() {
        mkdir -p $cache_dir
        cd $cache_dir

        git clone --depth 1 https://github.com/glfw/glfw.git
        cmake -S glfw -B glfw/build \
                -DGLFW_BUILD_WAYLAND=ON -DGLFW_BUILD_X11=ON
        cmake --build glfw/build -j "$(nproc)"
        sudo cmake --install glfw/build
}

# if cache is successfully restored, just install the builds
install_cached() {
        cd $cache_dir
        sudo cmake --install glfw/build
}

deps
if [ -d $cache_dir ]; then
        install_cached
else
        build_install
fi
