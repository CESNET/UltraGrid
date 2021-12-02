#!/bin/sh -eux
##
## Creates UltraGrid AppImage
##
## @param $1          (optional) zsync URL - location used for AppImage updater
## @env $appimage_key (optional) signing key
## @returns           name of created AppImage

APPDIR=UltraGrid.AppDir
APPPREFIX=$APPDIR/usr
ARCH=`uname -m`
DATE=`date +%Y%m%d`
GLIBC_VERSION=`ldd --version | sed -n '1s/.*\ \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p'`
APPNAME=UltraGrid-${DATE}.glibc${GLIBC_VERSION}-${ARCH}.AppImage
eval $(cat Makefile  | grep 'srcdir *=' | tr -d \  )

# redirect the whole output to stderr, output of this script is a created AppName only
(
exec 1>&2

mkdir tmpinstall $APPDIR
make DESTDIR=tmpinstall install
mv tmpinstall/usr/local $APPPREFIX

# add packet reflector
make -f $srcdir/hd-rum-multi/Makefile SRCDIR=$srcdir/hd-rum-multi
cp hd-rum $APPPREFIX/bin

# add platform and other Qt plugins if using dynamic libs
# @todo copy only needed ones
# @todo use https://github.com/probonopd/linuxdeployqt
PLUGIN_LIBS=
if [ -f $APPPREFIX/bin/uv-qt ]; then
        QT_DIR=$(dirname $(ldd $APPPREFIX/bin/uv-qt | grep Qt5Gui | grep -v found | awk '{ print $3 }'))
else
        QT_DIR=
fi
if [ -n "$QT_DIR" ]; then
        SRC_PLUGIN_DIR=$QT_DIR/qt5/plugins
        DST_PLUGIN_DIR=$APPPREFIX/lib/qt5/plugins
        mkdir -p $DST_PLUGIN_DIR
        cp -r $SRC_PLUGIN_DIR/* $DST_PLUGIN_DIR
        PLUGIN_LIBS=$(find $DST_PLUGIN_DIR -type f)

        # add DejaVu font
        mkdir $APPPREFIX/lib/fonts
        cp $(fc-list "DejaVu Sans" | sed 's/:.*//') $APPPREFIX/lib/fonts
fi

# copy dependencies
mkdir -p $APPPREFIX/lib
for n in $APPPREFIX/bin/* $APPPREFIX/lib/ultragrid/* $PLUGIN_LIBS; do
        for lib in `ldd $n | awk '{ print $3 }'`; do
                [ ! -f $lib ] || cp $lib $APPPREFIX/lib
        done
done

# Remove libraries that should not be bundled, see https://gitlab.com/probono/platformissues
wget https://raw.githubusercontent.com/probonopd/AppImages/master/excludelist
cat $(dirname $0)/excludelist.local >> excludelist
EXCLUDE_LIST=
while read -r x; do
        if [ "$x" = "" -o $(expr "x$x" : x\#) -eq 2 ]; then
                continue
        fi
        NAME=$(echo "$x" | awk '{ print $1 }')
        EXCLUDE_LIST="$EXCLUDE_LIST $NAME"
done < excludelist
for n in $EXCLUDE_LIST; do
        if [ -f $APPPREFIX/lib/$n ]; then
                rm $APPPREFIX/lib/$n
        fi
done

( cd $APPPREFIX/lib; rm -f libcmpto* ) # remove non-free components

# ship VA-API drivers if have libva
if [ -f $(echo $APPPREFIX/lib/libva.so.* | cut -d\  -f 1) ]; then
        for n in ${LIBVA_DRIVERS_PATH:-} /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri; do
                if [ -d "$n" ]; then
                        cp -r "$n" $APPPREFIX/lib/va
                        break
                fi
        done
fi

cp $srcdir/data/scripts/Linux-AppImage/AppRun $srcdir/data/ultragrid.png $APPDIR
cp $srcdir/data/uv-qt.desktop $APPDIR/ultragrid.desktop
APPIMAGEUPDATETOOL=$(command -v appimageupdatetool-x86_64.AppImage || true)
if [ -n "$APPIMAGEUPDATETOOL" ]; then
        cp $APPIMAGEUPDATETOOL $APPDIR/appimageupdatetool
else
        wget --no-verbose https://github.com/AppImage/AppImageUpdate/releases/download/continuous/appimageupdatetool-x86_64.AppImage -O $APPDIR/appimageupdatetool # use AppImageUpdate for GUI updater
fi
chmod ugo+x $APPDIR/appimageupdatetool

if [ -n "${appimage_key-}" ]; then
        echo "$appimage_key" >> key
        gpg --import key
        SIGN=--sign
fi

APPIMAGETOOL=$(command -v appimagetool-x86_64.AppImage || true)
if [ -z "$APPIMAGETOOL" ]; then
        wget --no-verbose https://github.com/AppImage/AppImageKit/releases/download/12/appimagetool-x86_64.AppImage -O appimagetool && chmod 755 appimagetool
        APPIMAGETOOL=./appimagetool
fi
UPDATE_INFORMATION=
if [ $# -ge 1 ]; then
        UPDATE_INFORMATION="-u zsync|$1"
fi
$APPIMAGETOOL ${SIGN+$SIGN }--comp gzip $UPDATE_INFORMATION $APPDIR $APPNAME

rm -rf $APPDIR tmpinstall
)

echo $APPNAME

