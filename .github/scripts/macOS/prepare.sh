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

import_signing_key() {
        if [ -z "$apple_key_p12_b64" ]; then
                return 0
        fi
        # Inspired by https://www.update.rocks/blog/osx-signing-with-travis/
        KEY_CHAIN=build.keychain
        KEY_CHAIN_PASS=build
        KEY_FILE=/tmp/signing_key.p12
        KEY_FILE_PASS=dummy
        echo "$apple_key_p12_b64" | base64 -d > $KEY_FILE
        security create-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN || true
        security default-keychain -s $KEY_CHAIN
        security unlock-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
        security import "$KEY_FILE" -A -P "$KEY_FILE_PASS"
        security set-key-partition-list -S apple-tool:,apple: -s -k $KEY_CHAIN_PASS $KEY_CHAIN
        printf '%b' "KEY_CHAIN_PASS=$KEY_CHAIN_PASS\nKEY_CHAIN=$KEY_CHAIN\n" \
                >> "$GITHUB_ENV"
}

export CPATH=/usr/local/include
export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s /usr/local/lib"
export LIBRARY_PATH=/usr/local/lib
if [ "$(uname -m)" = arm64 ]; then
        CPATH=/usr/local/include:/opt/homebrew/include
        DYLIBBUNDLER_FLAGS="$DYLIBBUNDLER_FLAGS -s /opt/homebrew/lib"
        LIBRARY_PATH="$LIBRARY_PATH:/opt/homebrew/lib"
        export LDFLAGS="-Wl,-rpath,/usr/local/lib"
        echo "LDFLAGS=$LDFLAGS" >> "$GITHUB_ENV"
fi
printf "%b" \
"CPATH=$CPATH\n\
LIBRARY_PATH=$LIBRARY_PATH\n" >> "$GITHUB_ENV"
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" >> "$GITHUB_ENV"
echo "/usr/local/opt/qt/bin" >> "$GITHUB_PATH"
echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> "$GITHUB_ENV"

import_signing_key

brew install autoconf automake libtool pkg-config \
        asciidoctor
brew install libsoxr speexdsp
brew install ffmpeg portaudio sdl2 sdl2_mixer sdl2_ttf
brew install molten-vk vulkan-headers
brew install imagemagick libcaca libnatpmp jack opencv wolfssl
brew install ossp-uuid # for cineform
brew install qt
brew install glm

mkdir $TEMP_INST
cd $TEMP_INST

"$srcroot/.github/scripts/macOS/install_dylibbundler_v2.sh"

# Install cross-platform deps
"$srcroot/.github/scripts/install-common-deps.sh"

"$srcroot/.github/scripts/macOS/install_others.sh"

# Remove installation files
cd
rm -rf $TEMP_INST

