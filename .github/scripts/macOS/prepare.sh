#!/bin/bash -eux

srcroot=$(cd "$(dirname "$0")/../../.."; pwd)
readonly srcroot

# shellcheck source=/dev/null
. "$srcroot/.github/scripts/json-common.sh"

TEMP_INST=/tmp/install

if [ -z "${GITHUB_ENV-}" ]; then
        GITHUB_ENV=/dev/null
        GITHUB_PATH=/dev/null
fi

export CPATH=/usr/local/include
export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s /usr/local/lib"
export LIBRARY_PATH=/usr/local/lib
if [ "$(uname -m)" = arm64 ]; then
        CPATH=/usr/local/include:/opt/homebrew/include
        DYLIBBUNDLER_FLAGS="$DYLIBBUNDLER_FLAGS -s /opt/homebrew/lib"
        LIBRARY_PATH="$LIBRARY_PATH:/opt/homebrew/lib"
fi
printf "%b" \
"CPATH=$CPATH\n\
LIBRARY_PATH=$LIBRARY_PATH\n" >> "$GITHUB_ENV"
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" >> "$GITHUB_ENV"
echo "/usr/local/opt/qt/bin" >> "$GITHUB_PATH"
echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> "$GITHUB_ENV"

brew install autoconf automake libtool pkg-config \
        asciidoctor
brew install libsoxr speexdsp
brew install ffmpeg portaudio sdl2 sdl2_mixer sdl2_ttf
brew install molten-vk vulkan-headers
brew install imagemagick libcaca libnatpmp jack opencv wolfssl
brew install ossp-uuid # for cineform
brew install qt
brew install glm
# TOREMOVE - missing header in Vulkan v1.3.264
if [ "$(brew info vulkan-headers | awk 'NR==1{print $4}')" = 1.3.264 ]; then
        sudo curl -L https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/main/\
include/vulkan/vulkan_hpp_macros.hpp -o /usr/local/include/vulkan/\
vulkan_hpp_macros.hpp
fi

mkdir $TEMP_INST
cd $TEMP_INST

"$srcroot/.github/scripts/macOS/install_dylibbundler_v2.sh"

# Install cross-platform deps
"$srcroot/.github/scripts/install-common-deps.sh"

"$srcroot/.github/scripts/macOS/install_others.sh"

# Remove installation files
cd
rm -rf $TEMP_INST

