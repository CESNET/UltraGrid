#!/bin/sh -eu
#
# Environment variables that may be updated by subsequent functions
# (eg. FEATURES) should not be set in a subshell, otherwise just the
# latter update will be written to $GITHUB_ENV. Also watch out early
# return from function not to be inside the subshell.

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
                curl -Sf -L "$XIMEA_DOWNLOAD_URL" -o $installer
        fi
        hdiutil mount $installer
        sudo cp -a /Volumes/XIMEA/m3api.framework \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
        sudo xattr -dr com.apple.quarantine \
                "$(xcrun --show-sdk-path)/System/Library/Frameworks/"
        umount /Volumes/XIMEA
)}

install_deltacast() {
        if [ ! "${SDK_URL-}" ]; then
                return
        fi
        tar xzf "$SDK_NONFREE_PATH/$DELTA_MAC_ARCHIVE"
        sudo cp -a Deltacast/Library/Frameworks/VideoMasterHD* \
                /Library/Frameworks/
        export FEATURES="${FEATURES+$FEATURES }--enable-deltacast"
        echo "FEATURES=$FEATURES" >> "$GITHUB_ENV"
        export COMMON_OSX_FLAGS="${COMMON_OSX_FLAGS+$COMMON_OSX_FLAGS }\
-F/Library/Frameworks"
        printf '%b' "COMMON_OSX_FLAGS=$COMMON_OSX_FLAGS\n" >> "$GITHUB_ENV"
}

install_glfw() {(
        git clone --depth 500 https://github.com/glfw/glfw.git
        cd glfw
        git am -3 "$srcroot"/.github/scripts/macOS/glfw-patches/*.patch
        cmake -DBUILD_SHARED_LIBS=ON .
        cmake --build . -j "$(sysctl -n hw.ncpu)"
        sudo cmake --install .
)}

install_libbacktrace() {(
        git clone --depth 1 https://github.com/ianlancetaylor/libbacktrace
        cd libbacktrace
        ./configure
        make -j "$(getconf NPROCESSORS_ONLN)"
        sudo make install
)}

# Install NDI
install_ndi() {(
        # installer downloaed by cache step
        installer=/private/var/tmp/Install_NDI_SDK_Apple.pkg
        sudo installer -pkg $installer -target /
        sudo mv /Library/NDI\ SDK\ for\ * /Library/NDI
)
        export CPATH=${CPATH:+"$CPATH:"}/Library/NDI/include
        printf '%b' "CPATH=$CPATH\n" >> "$GITHUB_ENV"
}

install_soundfont() {(
        sf_dir="$srcroot/data/template/macOS-bundle/Contents/share/soundfonts"
        mkdir -p "$sf_dir"
        cp "$GITHUB_WORKSPACE/data/default.sf3" "$sf_dir"
)}

install_syphon() {
        syphon_dst=/Library/Frameworks
(
        git clone --depth 1 https://github.com/Syphon/Syphon-Framework.git
        cd Syphon-Framework
        xcodebuild LD_DYLIB_INSTALL_NAME=$syphon_dst/\
Syphon.framework/Versions/A/Syphon
        sudo cp -R 'build/Release/Syphon.framework' \
                $syphon_dst
)
        export COMMON_OSX_FLAGS="${COMMON_OSX_FLAGS+$COMMON_OSX_FLAGS }\
-F$syphon_dst"
        printf '%b' "COMMON_OSX_FLAGS=$COMMON_OSX_FLAGS\n" >> "$GITHUB_ENV"
}

show_help=
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        show_help=1
fi

if [ $# -eq 0 ] || [ $show_help ]; then
        set -- deltacast glfw libbacktrace ndi soundfont syphon ximea
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
