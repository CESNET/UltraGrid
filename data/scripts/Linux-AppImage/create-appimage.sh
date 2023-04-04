#!/bin/sh -eux
##
## Creates UltraGrid AppImage
##
## @param $1          (optional) zsync URL - location used for AppImage updater
## @env $appimage_key (optional) signing key
## @returns           name of created AppImage

APPDIR=UltraGrid.AppDir
APPPREFIX=$APPDIR/usr
eval "$(grep 'srcdir *=' < Makefile | tr -d \  )"

# redirect the whole output to stderr, output of this script is a created AppName only
(
exec 1>&2

umask 022

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
QT_VER=
if [ -f $APPPREFIX/bin/uv-qt ]; then
        QT_LDD_DEP=$(ldd $APPPREFIX/bin/uv-qt | grep Qt.Gui | grep -v 'not found')
        QT_DIR=$(dirname "$(echo $QT_LDD_DEP | awk '{ print $3 }')")
        QT_VER=$(echo $QT_LDD_DEP | awk '{ print $1  }' | sed 's/.*Qt\([0-9]\)Gui.*/\1/g')
else
        QT_DIR=
fi
if [ -n "$QT_DIR" ]; then
        QT_PLUGIN_PREFIX=$QT_DIR/qt$QT_VER
        SRC_PLUGIN_DIR=$QT_PLUGIN_PREFIX/plugins
        DST_PLUGIN_DIR=$APPPREFIX/lib/$(basename "$QT_PLUGIN_PREFIX")/plugins
        mkdir -p "$DST_PLUGIN_DIR"
        cp -r "$SRC_PLUGIN_DIR"/* "$DST_PLUGIN_DIR"
        PLUGIN_LIBS=$(find "$DST_PLUGIN_DIR" -type f)
fi

add_fonts() { # for GUI+testcard2
        # add DejaVu font
        mkdir $APPPREFIX/share/fonts
        for family in "DejaVu Sans" "DejaVu Sans Mono"; do
                for style in "Book" "Bold"; do
                        FONT_PATH=$(fc-match "$family:style=$style" file | sed 's/.*=//')
                        cp "$FONT_PATH" $APPPREFIX/share/fonts
                done
        done
}
add_fonts

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

if command wget >/dev/null && wget -V | grep -q https; then
        dl='wget -O -'
elif command -v curl >/dev/null; then
        dl='curl -L'
fi

# Remove libraries that should not be bundled, see https://gitlab.com/probono/platformissues
[ -f excludelist ] || $dl https://raw.githubusercontent.com/probonopd/AppImages/master/excludelist > excludelist || exit 1
DIRNAME=$(dirname "$0")
uname_m=$(uname -m)
excl_list_arch=x86
if expr "$uname_m" : arm >/dev/null || expr "uname_m" : aarch64; then
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
        if [ "$n" = libjack.so.0 ]; then # JACK is currently handled in AppRun
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
        $dl https://github.com/AppImage/AppImageUpdate/releases/download/continuous/appimageupdatetool-x86_64.AppImage > $appimageupdatetool # use AppImageUpdate for GUI updater
fi
cp "$appimageupdatetool" $APPDIR/appimageupdatetool
chmod ugo+x $APPDIR/appimageupdatetool
if [ -f /lib/x86_64-linux-gnu/libfuse.so.2 ]; then
        mkdir $APPDIR/appimageupdatetool-lib
        cp /lib/x86_64-linux-gnu/libfuse.so.2 $APPDIR/appimageupdatetool-lib
fi

GIT_ROOT=$(git rev-parse --show-toplevel || true)
if [ -n "${appimage_key-}" ] && [ -n "${GIT_ROOT-}" ]; then
        echo "$appimage_key" >> "$GIT_ROOT/pubkey.asc"
fi

mkappimage=$(command -v ./mkappimage || command -v mkappimage-x86_64.AppImage || command -v mkappimage || true)
if [ -z "$mkappimage" ]; then
        mkai_url=$($dl https://api.github.com/repos/probonopd/go-appimage/releases/tags/continuous | grep "browser_download_url.*mkappimage-.*-x86_64.AppImage" | head -n 1 | cut -d '"' -f 4)
        $dl "$mkai_url" > mkappimage
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
# shellcheck disable=SC2086 # word spliting of $UPDATE_INFORMATION is requested behavior
$mkappimage $UPDATE_INFORMATION $APPDIR

rm -rf $APPDIR tmpinstall
)

