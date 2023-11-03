#!/bin/bash -eux

srcroot=$(cd "$(dirname "$0")/../../.."; pwd)
readonly srcroot

# shellcheck source=/dev/null
. "$srcroot/.github/scripts/json-common.sh"

TEMP_INST=/tmp/install

if [ -z "${GITHUB_ENV-}" ]; then
        GITHUB_ENV=/dev/null
fi

CPATH=/usr/local/include
DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s /usr/local/lib"
LIBRARY_PATH=/usr/local/lib
# shellcheck disable=SC2140
printf "%b" \
"CPATH=$CPATH\n"\
"LIBRARY_PATH=$LIBRARY_PATH\n" >> "$GITHUB_ENV"
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" >> "$GITHUB_ENV"
echo "/usr/local/opt/qt/bin" >> "$GITHUB_PATH"
echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> "$GITHUB_ENV"

brew install autoconf automake libtool pkg-config
brew install libsoxr speexdsp
brew install ffmpeg portaudio sdl2 sdl2_mixer sdl2_ttf
brew install molten-vk vulkan-headers
brew install imagemagick libcaca libnatpmp jack opencv wolfssl
brew install ossp-uuid # for cineform
brew install qt
brew install glm

.github/scripts/macOS/install_dylibbundler_v2.sh

mkdir $TEMP_INST
cd $TEMP_INST

# Install XIMEA (see <dmg>/install.app/Contents/MacOS/install.sh)
install_ximea() {
        hdiutil mount /private/var/tmp/XIMEA_OSX_SP.dmg
        sudo cp -a /Volumes/XIMEA/m3api.framework "$(xcrun --show-sdk-path)/System/Library/Frameworks"
        sudo xattr -dr com.apple.quarantine "$(xcrun --show-sdk-path)/System/Library/Frameworks"
        umount /Volumes/XIMEA
}

install_aja() {
        # shellcheck source=/dev/null
        . "$srcroot/.github/scripts/aja-common.sh"
        AJA_DIRECTORY=/private/var/tmp/ntv2sdk
        git clone --depth 1 https://github.com/aja-video/ntv2 $AJA_DIRECTORY
        cd $AJA_DIRECTORY
        echo "AJA_DIRECTORY=$AJA_DIRECTORY" >> "$GITHUB_ENV"
        download_aja_release_asset libs_mac_ aja_build.tar.gz
        tar xzf aja_build.tar.gz
        sudo cp Release/x64/* /usr/local/lib
        cd $TEMP_INST
}

install_deltacast() {
        DELTA_CACHE_INST=${SDK_NONFREE_PATH-nonexistent}/VideoMasterHD_inst
        if [ ! -d "$DELTA_CACHE_INST" ]; then
                return 0
        fi
        FEATURES="$FEATURES --enable-deltacast"
        echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
        sudo cp -a "$DELTA_CACHE_INST"/* "$(xcrun --show-sdk-path)/System/Library/Frameworks"
}

install_glfw() {(
        git clone --depth 500 https://github.com/glfw/glfw.git
        cd glfw
        git fetch --depth 500 https://github.com/MartinPulec/glfw.git
        git merge FETCH_HEAD
        cmake -DBUILD_SHARED_LIBS=ON .
        cmake --build . --parallel
        sudo cmake --install .
)}

# Install NDI
install_ndi() {
        sudo installer -pkg /private/var/tmp/Install_NDI_SDK_Apple.pkg -target /
        sudo mv /Library/NDI\ SDK\ for\ * /Library/NDI
        sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' < /Library/NDI/Version.txt | sudo tee /usr/local/include/ndi_version.h
        if [ -d /Library/NDI/lib/x64 ]; then # NDI 4
                cd /Library/NDI/lib/x64
                sudo ln -s libndi.?.dylib libndi.dylib
                NDI_LIB=/Library/NDI/lib/x64
        else # NDI 5
                NDI_LIB=/Library/NDI/lib/macOS
        fi
        export CPATH=${CPATH:+"$CPATH:"}/Library/NDI/include
        export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s $NDI_LIB"
        export LIBRARY_PATH=${LIBRARY_PATH:+"$LIBRARY_PATH:"}$NDI_LIB
        export MY_DYLD_LIBRARY_PATH="${MY_DYLD_LIBRARY_PATH:+$MY_DYLD_LIBRARY_PATH:}$NDI_LIB"
        printf '%b' "CPATH=$CPATH\nDYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS\nLIBRARY_PATH=$LIBRARY_PATH\nMY_DYLD_LIBRARY_PATH=$MY_DYLD_LIBRARY_PATH\n" >> "$GITHUB_ENV"
        cd $TEMP_INST
}

install_live555() {
        git clone https://github.com/xanview/live555/
        cd live555
        git checkout 35c375
        ./genMakefiles macosx
        make -j "$(sysctl -n hw.ncpu)" install
        cd ..
}

install_soundfont() {
        . "$srcroot/.github/scripts/defs.sh"
        sf_dir="$srcroot/data/MacOS-bundle-template/Contents/share/soundfonts"
        mkdir -p "$sf_dir"
        curl -L "$DEFAULT_SF_URL" -o "$sf_dir/default.${DEFAULT_SF_URL##*.}"
}

install_syphon() {
        curl -LO https://github.com/Syphon/Syphon-Framework/releases/download/5/Syphon.SDK.5.zip
        unzip Syphon.SDK.5.zip
        sudo cp -R 'Syphon SDK 5/Syphon.framework' /Library/Frameworks
}

# Install cross-platform deps
"$srcroot/.github/scripts/install-common-deps.sh"

install_aja
install_deltacast
install_glfw
install_live555
install_ndi
install_soundfont
install_syphon
install_ximea

# Remove installation files
cd
rm -rf $TEMP_INST

