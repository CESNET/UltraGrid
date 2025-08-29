#!/bin/sh -eux
##
## Creates UltraGrid AppImage
##
## @param $1          (optional) zsync URL - location used for AppImage updater
## @env $appimage_key (optional) signing key
## (new method, see https://github.com/probonopd/go-appimage/issues/318)
## \n
## Contains base64-encoded .tar.gz of pubkey.asc+privkey.asc.enc files
## containing exported GPG public and private (encrypted) key.
## \n
## Private key is encrypted by OpenSSL (see appimagetool.go source):
## `openssl aes-256-cbc -pass pass:dummy -in privkey.asc -out privkey.asc.enc -a -md sha256`
## @returns           name of created AppImage

APPDIR=UltraGrid.AppDir
APPPREFIX=$APPDIR/usr
eval "$(grep 'srcdir *=' < Makefile | tr -d \  )"

umask 022

# soft fail - fail in CI, otherwise continue
handle_error() {
        red=1
        tput setaf $red 2>/dev/null || true
        tput bold 2>/dev/null || true
        echo "$1" >&2
        tput sgr0 2>/dev/null || true
        if [ -n "${GITHUB_REPOSITORY:-}" ]; then
                exit 2
        fi
}

mkdir tmpinstall $APPDIR
make DESTDIR=tmpinstall install
mv tmpinstall/usr/local $APPPREFIX

# add packet reflector
make -f "${srcdir?srcdir not found}/hd-rum-multi/Makefile" "SRCDIR=$srcdir/hd-rum-multi"
cp hd-rum $APPPREFIX/bin
make -C "$srcdir/tools" convert
cp "$srcdir/tools/convert" $APPPREFIX/bin

# add platform and other Qt plugins if using dynamic libs
# @todo copy only needed ones
# @todo use https://github.com/probonopd/linuxdeployqt
PLUGIN_LIBS=
QT_DIR=
QT_VER=
if [ -f $APPPREFIX/bin/uv-qt ]; then
        QT_LDD_DEP=$(ldd $APPPREFIX/bin/uv-qt | grep Qt.Gui | grep -v 'not found')
        QT_DIR=$(echo "$QT_LDD_DEP" | awk '{ print $3 }')
        QT_DIR=$(dirname "$QT_DIR")
        QT_VER=$(echo "$QT_LDD_DEP" | awk '{ print $1  }' | sed 's/.*Qt\([0-9]\)Gui.*/\1/g')
fi
if [ -n "$QT_DIR" ]; then
        QT_PLUGIN_PREFIX=$QT_DIR/qt$QT_VER
        SRC_PLUGIN_DIR=$QT_PLUGIN_PREFIX/plugins
        DST_PLUGIN_DIR=$APPPREFIX/lib/$(basename "$QT_PLUGIN_PREFIX")/plugins
        mkdir -p "$DST_PLUGIN_DIR"
        cp -r "$SRC_PLUGIN_DIR"/* "$DST_PLUGIN_DIR"
        PLUGIN_LIBS=$(find "$DST_PLUGIN_DIR" -type f)
fi

if [ -f $APPPREFIX/lib/ultragrid/ultragrid_vo_pp_text.so ]; then
        if ! command -v convert >/dev/null; then
                handle_error 'IM convert missing! (needed for bundle)'
        fi
        # https://stackoverflow.com/a/53355763
        conf_path=$(convert -list configure |
                sed -n '/CONFIGURE_PATH/ { s/[A-Z_]* *//; p; q; }')
        codr_path=$(convert -list configure |
                sed -n '/CODER_PATH/ { s/[A-Z_]* *//; p; q; }')
        filt_path=$(convert -list configure |
                sed -n '/FILTER_PATH/ { s/[A-Z_]* *//; p; q; }')
        mkdir $APPDIR/etc $APPPREFIX/share/IM
        cp -r "$conf_path" $APPDIR/etc/IM
        cp -r "$codr_path" $APPPREFIX/share/IM/coders
        cp -r "$filt_path" $APPPREFIX/share/IM/filters
fi

add_fonts() { # for GUI+testcard2
        if ! command -v fc-match >/dev/null; then
                handle_error "fc-match not found, not copying fonts!"
                return
        fi
        # add DejaVu font
        mkdir $APPPREFIX/share/fonts
        for family in "DejaVu Sans" "DejaVu Sans Mono"; do
                for style in "Book" "Bold"; do
                        FONT_PATH=$(fc-match "$family:style=$style" file | sed 's/.*=//')
                        cp "$FONT_PATH" $APPPREFIX/share/fonts
                done
        done
        if ls $APPPREFIX/lib/*mixer* >/dev/null 2>&1 ||
           ls $APPPREFIX/lib/ultragrid/*fluidsynth* >/dev/null 2>&1; then
                mkdir -p $APPPREFIX/share/soundfonts
                cp "$srcdir/data/default.sf3" $APPPREFIX/share/soundfonts/
        fi
}

# copy dependencies
mkdir -p $APPPREFIX/lib
for n in "$APPPREFIX"/bin/* "$APPPREFIX"/lib/ultragrid/* $PLUGIN_LIBS; do
        for lib in $(ldd "$n" | awk '{ print $3 }'); do
                [ ! -f "$lib" ] && continue
                DST_NAME=$APPPREFIX/lib/$(basename "$lib")
                [ -f "$DST_NAME" ] && continue
                cp "$lib" $APPPREFIX/lib
        done
done

# hide Wayland libraries
if ls $APPPREFIX/lib/libwayland-* >/dev/null 2>&1; then
        mkdir $APPPREFIX/lib/wayland
        mv $APPPREFIX/lib/libwayland-* $APPPREFIX/lib/wayland
fi

add_fonts

if command -v curl >/dev/null; then
        dl() {
                curl --fail -sSL ${GITHUB_TOKEN+-H "Authorization: token $GITHUB_TOKEN"} "$1"
        }
elif command -v wget >/dev/null && wget -V | grep -q https; then
        dl() {
                wget -O - ${GITHUB_TOKEN+--header "Authorization: token $GITHUB_TOKEN"} "$1"
        }
else
        echo "Neither wget nor curl was found - if one needed later, it will " \
                "fail!" >&2
fi

# Remove libraries that should not be bundled, see https://gitlab.com/probono/platformissues
[ -f excludelist ] || dl https://raw.githubusercontent.com/probonopd/AppImages/master/excludelist > excludelist || exit 1
DIRNAME=$(dirname "$0")
uname_m=$(uname -m)
excl_list_arch=x86
if expr "$uname_m" : arm >/dev/null || expr "$uname_m" : aarch64 > /dev/null; then
        excl_list_arch=arm
fi
cat "$DIRNAME/excludelist.local.$excl_list_arch" >> excludelist
EXCLUDE_LIST=
while read -r x; do
        if [ -z "$x" ] || expr "x$x" : x\# > /dev/null; then
                continue
        fi
        NAME=$(echo "$x" | awk '{ print $1 }')
        EXCLUDE_LIST="$EXCLUDE_LIST $NAME"
done < excludelist
for n in $EXCLUDE_LIST; do
        # these dependencies preloaded by AppRun if found in system - include
        # them for the cases when isn't
        if [ "$n" = libjack.so.0 ] || [ "$n" = libpipewire-0.3.so.0 ]; then
                continue
        fi
        if [ -f "$APPPREFIX/lib/$n" ]; then
                rm "$APPPREFIX/lib/$n"
        fi
done

( cd $APPPREFIX/lib; rm -f libcmpto* ) # remove non-free components

# ship VA-API drivers if have libva
if [ -f "$(echo $APPPREFIX/lib/libva.so.* | cut -d\  -f 1)" ]; then
        for n in ${LIBVA_DRIVERS_PATH:-} /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri; do
                if [ -d "$n" ]; then
                        cp -r "$n" $APPPREFIX/lib/va
                        break
                fi
        done
fi

cp -r "$srcdir/data/scripts/Linux-AppImage/AppRun" "$srcdir/data/scripts/Linux-AppImage/scripts" "$srcdir/data/ultragrid.png" $APPDIR
cp "$srcdir/data/uv-qt.desktop" $APPDIR/cz.cesnet.ultragrid.desktop
appimageupdatetool=$(command -v appimageupdatetool-x86_64.AppImage || command -v ./appimageupdatetool || true)
if [ -z "$appimageupdatetool" ]; then
        appimageupdatetool=./appimageupdatetool
        dl https://github.com/AppImage/AppImageUpdate/releases/download/continuous/appimageupdatetool-x86_64.AppImage > $appimageupdatetool # use AppImageUpdate for GUI updater
fi
cp "$appimageupdatetool" $APPDIR/appimageupdatetool
chmod ugo+x $APPDIR/appimageupdatetool
if [ -f /lib/x86_64-linux-gnu/libfuse.so.2 ]; then
        mkdir $APPDIR/appimageupdatetool-lib
        cp /lib/x86_64-linux-gnu/libfuse.so.2 $APPDIR/appimageupdatetool-lib
fi

# TODO: temporarily (? 2025-01-25) disable signing because validation fails
unset appimage_key
GIT_ROOT=$(git rev-parse --show-toplevel || true)
if [ -n "${appimage_key-}" ] && [ -n "${GIT_ROOT-}" ]; then
        echo "$appimage_key" | base64 -d | tar -C "$GIT_ROOT" -xzaf -
        export super_secret_password=dummy
fi

mkappimage=$(command -v ./mkappimage || command -v mkappimage-x86_64.AppImage || command -v mkappimage || true)
if [ -z "$mkappimage" ]; then
        mkai_url=$(dl https://api.github.com/repos/probonopd/go-appimage/releases/tags/continuous | grep "browser_download_url.*mkappimage-.*-x86_64.AppImage" | head -n 1 | cut -d '"' -f 4)
        dl "$mkai_url" > mkappimage
        chmod +x mkappimage
        mkappimage=./mkappimage
fi
if "$mkappimage" 2>&1 | grep fuse; then
        if [ ! -d mkappimage-extracted ]; then
                "$mkappimage" --appimage-extract
                mv squashfs-root mkappimage-extracted
        fi
        mkappimage="mkappimage-extracted/AppRun"
fi

UPDATE_INFORMATION=
if [ $# -ge 1 ]; then
        UPDATE_INFORMATION="-u zsync|$1"
fi
# shellcheck disable=SC1007,SC2086 # word spliting of
# $UPDATE_INFORMATION is a requested behavior
GITHUB_TOKEN= $mkappimage $UPDATE_INFORMATION $APPDIR

rm -rf $APPDIR tmpinstall

