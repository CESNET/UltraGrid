#!/bin/sh -eu

srcroot=${GITHUB_WORKSPACE-$PWD}
readonly srcroot

if [ -z "${GITHUB_ENV-}" ]; then
        GITHUB_ENV=/dev/null
fi

# shellcheck source=/dev/null
. "$srcroot/.github/scripts/json-common.sh"

# Install XIMEA (see <dmg>/install.app/Contents/MacOS/install.sh)
install_ximea() {(
        installer=/private/var/tmp/XIMEA_OSX_SP.dmg
        if [ ! -f $installer ]; then
                curl -S -L https://www.ximea.com/downloads/recent/XIMEA_OSX_SP\
.dmg -o $installer
        fi
        hdiutil mount $installer
        sudo cp -a /Volumes/XIMEA/m3api.framework \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
        sudo xattr -dr com.apple.quarantine \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
        umount /Volumes/XIMEA
)}

install_aja() {(
        AJA_DIRECTORY=/private/var/tmp/ntv2sdk
        git clone --depth 1 https://github.com/aja-video/ntv2 $AJA_DIRECTORY
        cd $AJA_DIRECTORY
        echo "AJA_DIRECTORY=$AJA_DIRECTORY" >>"$GITHUB_ENV"
        "$srcroot/.github/scripts/download-gh-asset.sh" aja-video/ntv2 \
                libs_mac_ aja_build.tar.gz
        tar xzf aja_build.tar.gz
        arch_subd=$(uname -m | sed 's/x86_64/x64/')
        sudo cp Release/"$arch_subd"/* /usr/local/lib/
)}

install_deltacast() {(
        DELTA_CACHE_INST=${SDK_NONFREE_PATH-nonexistent}/VideoMasterHD_inst
        if [ ! -d "$DELTA_CACHE_INST" ] || [ "$(uname -m)" = arm64 ]; then
                return 0
        fi
        FEATURES="$FEATURES --enable-deltacast"
        echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
        sudo cp -a "$DELTA_CACHE_INST"/* \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
)}

install_glfw() {(
        git clone --depth 500 https://github.com/glfw/glfw.git
        cd glfw
        git fetch --depth 500 https://github.com/MartinPulec/glfw.git
        git merge FETCH_HEAD
        cmake -DBUILD_SHARED_LIBS=ON .
        cmake --build . -j "$(sysctl -n hw.ncpu)"
        sudo cmake --install .
)}

# Install NDI
install_ndi() {(
        installer=/private/var/tmp/Install_NDI_SDK_Apple.pkg
        if [ ! -f $installer ]; then
                curl -L https://downloads.ndi.tv/SDK/NDI_SDK_Mac/Install_NDI_\
SDK_v5_Apple.pkg -o /private/var/tmp/Install_NDI_SDK_Apple.pkg
        fi
        sudo installer -pkg $installer -target /
        sudo mv /Library/NDI\ SDK\ for\ * /Library/NDI
        sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' < /Library/NDI/Version.txt |
                sudo tee /usr/local/include/ndi_version.h
        if [ -d /Library/NDI/lib/x64 ]; then # NDI 4
                cd /Library/NDI/lib/x64
                sudo ln -s libndi.?.dylib libndi.dylib
                NDI_LIB=/Library/NDI/lib/x64
        else # NDI 5
                NDI_LIB=/Library/NDI/lib/macOS
        fi
        export CPATH=${CPATH:+"$CPATH:"}/Library/NDI/include
        export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }\
-s $NDI_LIB"
        export LIBRARY_PATH=${LIBRARY_PATH:+"$LIBRARY_PATH:"}$NDI_LIB
        export MY_DYLD_LIBRARY_PATH="${MY_DYLD_LIBRARY_PATH:+\
$MY_DYLD_LIBRARY_PATH:}$NDI_LIB"
        printf '%b' "CPATH=$CPATH\nDYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS\n\
LIBRARY_PATH=$LIBRARY_PATH\nMY_DYLD_LIBRARY_PATH=$MY_DYLD_LIBRARY_PATH\n" >> \
"$GITHUB_ENV"
)}

install_live555() {(
        git clone https://github.com/xanview/live555/
        cd live555
        git checkout 35c375
        ./genMakefiles macosx
        sudo make -j "$(sysctl -n hw.ncpu)" install
)}

install_soundfont() {(
        . "$srcroot/.github/scripts/defs.sh"
        sf_dir="$srcroot/data/MacOS-bundle-template/Contents/share/soundfonts"
        mkdir -p "$sf_dir"
        curl -L "$DEFAULT_SF_URL" -o "$sf_dir/default.${DEFAULT_SF_URL##*.}"
)}

install_syphon() {(
        if [ "$(uname -m)" = arm64 ]; then
                git clone --depth 1 https://github.com/Syphon/\
Syphon-Framework.git
                cd Syphon-Framework
                xcodebuild LD_DYLIB_INSTALL_NAME=@executable_path/../\
Frameworks/Syphon.framework/Versions/A/Syphon
                cd build/Release
        else
                curl -LO https://github.com/Syphon/Syphon-Framework/releases/\
download/5/Syphon.SDK.5.zip
                unzip Syphon.SDK.5.zip
                cd 'Syphon SDK 5'
        fi
        sudo cp -R 'Syphon.framework' \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
)}

show_help=
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        show_help=1
fi

if [ $# -eq 0 ] || [ $show_help ]; then
        set --  aja deltacast glfw live555 ndi soundfont syphon ximea
fi

if [ $show_help ]; then
        printf "Usage:\n"
        printf "\t%s [<features>] | [ -h | --help | help ]\n" "$0"
        printf "\nInstall all aditional dependencies (without arguments) or \
install one explicitly.\n"
        printf "\nAvailable ones: %s%s%s\n" "$(tput bold)" "$*" "$(tput sgr0)"
        exit 0
fi

set -x
while [ $# -gt 0 ]; do
        install_"$1"
        shift
done
