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
export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }\
 -s /usr/local/lib -s /Library/Frameworks"
export LDFLAGS="-Wl,-rpath,/usr/local/lib -Wl,-rpath,/Library/Frameworks"
export LIBRARY_PATH=/usr/local/lib
if [ "$(uname -m)" = arm64 ]; then
        CPATH=/usr/local/include:/opt/homebrew/include
        DYLIBBUNDLER_FLAGS="$DYLIBBUNDLER_FLAGS -s /opt/homebrew/lib"
        LIBRARY_PATH="$LIBRARY_PATH:/opt/homebrew/lib"
fi
printf "%b" "\
CPATH=$CPATH\n\
LDFLAGS=$LDFLAGS\n\
LIBRARY_PATH=$LIBRARY_PATH\n" >> "$GITHUB_ENV"
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" >> "$GITHUB_ENV"
echo "/usr/local/opt/qt/bin" >> "$GITHUB_PATH"
echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> "$GITHUB_ENV"

set -- \
        asciidoctor \
        autoconf \
        automake \
        ffmpeg \
        fluidsynth \
        glm \
        imagemagick \
        jack \
        libcaca \
        libnatpmp \
        libsoxr \
        libtool \
        molten-vk \
        opencv \
        ossp-uuid `#for cineform` \
        pkg-config \
        portaudio \
        qt \
        sdl3 \
        sdl3_ttf \
        speexdsp \
        vulkan-headers \
        wolfssl \

# shellcheck disable=SC2034
for n in $(seq $#); do
        # if not installed, add on the back of positional parameters
        if ! brew list "$1" >/dev/null 2>&1; then
                set -- "$@" "$1"
        fi
        shift # remove from the front
done

brew install "$@"
 
mkdir $TEMP_INST
cd $TEMP_INST

"$srcroot/.github/scripts/macOS/install_dylibbundler_v2.sh"

# Install cross-platform deps
"$srcroot/.github/scripts/install-common-deps.sh"

"$srcroot/.github/scripts/macOS/install_others.sh"

# Remove installation files
cd
rm -rf $TEMP_INST

