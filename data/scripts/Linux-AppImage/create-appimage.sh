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

# add platform and other Qt plugins if using dynamic libs
# @todo copy only needed ones
# @todo use https://github.com/probonopd/linuxdeployqt
PLUGIN_LIBS=
if [ -f $APPPREFIX/bin/uv-qt ]; then
        QT_DIR=$(dirname "$(ldd $APPPREFIX/bin/uv-qt | grep Qt.Gui | grep -v 'not found' | awk '{ print $3 }')")
else
        QT_DIR=
fi
if [ -n "$QT_DIR" ]; then
        QT_PLUGIN_PREFIX=$(find "$QT_DIR" -maxdepth 1 -type d -name 'qt?')
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

# Remove libraries that should not be bundled, see https://gitlab.com/probono/platformissues
wget https://raw.githubusercontent.com/probonopd/AppImages/master/excludelist
DIRNAME=$(dirname "$0")
cat "$DIRNAME/excludelist.local" >> excludelist
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
APPIMAGEUPDATETOOL=$(command -v appimageupdatetool-x86_64.AppImage || true)
if [ -n "$APPIMAGEUPDATETOOL" ]; then
        cp "$APPIMAGEUPDATETOOL" $APPDIR/appimageupdatetool
else
        wget --no-verbose https://github.com/AppImage/AppImageUpdate/releases/download/continuous/appimageupdatetool-x86_64.AppImage -O $APPDIR/appimageupdatetool # use AppImageUpdate for GUI updater
fi
chmod ugo+x $APPDIR/appimageupdatetool

GIT_ROOT=$(git rev-parse --show-toplevel || true)
if [ -n "${appimage_key-}" ] && [ -n "${GIT_ROOT-}" ]; then
        echo "$appimage_key" >> "$GIT_ROOT/pubkey.asc"
fi

MKAPPIMAGE=$(command -v mkappimage-x86_64.AppImage || true)
if [ -z "$MKAPPIMAGE" ]; then
        MKAI_PATH=$(curl -s https://api.github.com/repos/probonopd/go-appimage/releases/tags/continuous | grep "browser_download_url.*mkappimage-.*-x86_64.AppImage" | head -n 1 | cut -d '"' -f 4)
        wget -q -c "$MKAI_PATH" -O mkappimage
        chmod +x mkappimage
        MKAPPIMAGE=./mkappimage
fi
UPDATE_INFORMATION=
if [ $# -ge 1 ]; then
        UPDATE_INFORMATION="-u zsync|$1"
fi
# shellcheck disable=SC2086 # word spliting of $UPDATE_INFORMATION is requested behavior
$MKAPPIMAGE $UPDATE_INFORMATION $APPDIR

rm -rf $APPDIR tmpinstall
)

