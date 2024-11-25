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

# TOREMOVE: temporal CI fix, remove after a short period (let say after 2024)
for n in $(brew list --formula -1 | grep -E '^(pkg-config(@.*)?|pkgconf)$'); do
        brew uninstall "$n"
done
brew install pkg-config
# if pkg-config is not alias for pkgconf, install it for deps but unlink
if brew list pkg-config | grep -qv pkgconf; then
        brew uninstall pkg-config
        brew install pkgconf
        brew unlink pkgconf
        brew install pkg-config
fi

set -- \
        asciidoctor \
        autoconf \
        automake \
        ffmpeg \
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
        pkgconf \
        portaudio \
        qt \
        sdl2 \
        sdl2_mixer \
        sdl2_ttf \
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

