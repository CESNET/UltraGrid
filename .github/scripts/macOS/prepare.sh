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

brew_install_list=
# if not installed, add on the back of brew_install_list
# the rationale is not to install already installed items, otherwise
# the Homebrew issues warning about it which is passed to GH Action
# warning, which is ugly
brew_add() {
        if ! brew list "$1" >/dev/null 2>&1; then
                brew_install_list="$brew_install_list $1"
        fi
}
brew_add asciidoctor
brew_add autoconf
brew_add automake
brew_add ffmpeg
brew_add fluidsynth
brew_add glm
brew_add imagemagick
brew_add jack
brew_add libcaca
brew_add libnatpmp
brew_add libsoxr
brew_add libtool
brew_add molten-vk
brew_add opencv
brew_add ossp-uuid # for cineform
brew_add pkg-config
brew_add portaudio
brew_add qt
brew_add sdl3
brew_add sdl3_ttf
brew_add speexdsp
brew_add vulkan-headers
brew_add wolfssl
# shellcheck disable=SC2086
brew install $brew_install_list

mkdir $TEMP_INST
cd $TEMP_INST

HOMEBREW_PREFIX=$(brew --prefix)
sed '/^Requires\.private/ s/^/# /'\
 "$HOMEBREW_PREFIX"/lib/pkgconfig/fluidsynth.pc > override
sudo mv override "$HOMEBREW_PREFIX"/lib/pkgconfig/fluidsynth.pc

"$srcroot/.github/scripts/macOS/install_dylibbundler_v2.sh"

# Install cross-platform deps
"$srcroot/.github/scripts/install-common-deps.sh"

"$srcroot/.github/scripts/macOS/install_others.sh"

# TODO TOREMOVE when no longer needed
# fix incorrectly included '$ORIGIN' in rpath causing dylibbundler fail
if [ "$(uname -m)" = arm64 ]; then
        sudo install_name_tool -delete_rpath '$ORIGIN' /opt/homebrew/lib/libtbb.dylib || true
fi

# Remove installation files
cd
rm -rf $TEMP_INST

