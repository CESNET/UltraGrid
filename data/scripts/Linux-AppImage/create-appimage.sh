#!/bin/sh -eux

APPDIR=UltraGrid.AppDir
ARCH=`uname -m`
DATE=`date +%Y%m%d`
GLIBC_VERSION=`ldd --version | sed -n '1s/.*\ \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p'`
APPNAME=UltraGrid-${DATE}.glibc${GLIBC_VERSION}-${ARCH}.AppImage

# redirect the whole output to stderr, output of this script is a created AppName only
(
exec 1>&2

mkdir tmpinstall
make DESTDIR=tmpinstall install
mv tmpinstall/usr/local $APPDIR

# add platform and other Qt plugins if using dynamic libs
# @todo copy only needed ones
# @todo use https://github.com/probonopd/linuxdeployqt
PLUGIN_LIBS=
QT_DIR=$(dirname $(ldd $APPDIR/bin/uv-qt | grep Qt5Gui | grep -v found | awk '{ print $3 }'))
if [ -f $APPDIR/bin/uv-qt -a -n $QT_DIR ]; then
        SRC_PLUGIN_DIR=$QT_DIR/qt5/plugins
        DST_PLUGIN_DIR=$APPDIR/lib/qt5/plugins
        mkdir -p $DST_PLUGIN_DIR
        cp -r $SRC_PLUGIN_DIR/* $DST_PLUGIN_DIR
        PLUGIN_LIBS=$(find $DST_PLUGIN_DIR -type f)
fi

for n in $APPDIR/bin/* $APPDIR/lib/ultragrid/* $PLUGIN_LIBS; do
        for lib in `ldd $n | awk '{ print $3 }'`; do
                [ ! -f $lib ] || cp $lib $APPDIR/lib
        done
done

mkdir $APPDIR/lib/fonts
cp $(fc-list "DejaVu Sans" | sed 's/:.*//') $APPDIR/lib/fonts

# Remove libraries that should not be bundled, see https://gitlab.com/probono/platformissues
wget https://raw.githubusercontent.com/probonopd/AppImages/master/excludelist
EXCLUDE_LIST=
while read -r x; do
        if [ "$x" = "" -o $(expr "x$x" : x\#) -eq 2 ]; then
                continue
        fi
        NAME=$(echo "$x" | awk '{ print $1 }')
        if [ "$NAME" = libjack.so.0 ]; then # JACK is currently handled in AppRun
                continue
        fi
        EXCLUDE_LIST="$EXCLUDE_LIST $NAME"
done < excludelist
for n in $EXCLUDE_LIST; do
        if [ -f $APPDIR/lib/$n ]; then
                rm $APPDIR/lib/$n
        fi
done

( cd $APPDIR/lib; rm -f libcmpto* ) # remove non-free components

cp data/scripts/Linux-AppImage/AppRun data/scripts/Linux-AppImage/uv-wrapper.sh data/ultragrid.png $APPDIR
cp data/uv-qt.desktop $APPDIR/ultragrid.desktop
wget --no-verbose https://github.com/AppImage/AppImageUpdate/releases/download/continuous/AppImageUpdate-x86_64.AppImage -O $APPDIR/appimageupdatetool
chmod ugo+x $APPDIR/appimageupdatetool

if [ -n "$appimage_key" ]; then
        echo "$appimage_key" >> key
        gpg --import key
fi

wget --no-verbose https://github.com/AppImage/AppImageKit/releases/download/12/appimagetool-x86_64.AppImage -O appimagetool && chmod 755 appimagetool
UPDATE_INFORMATION=
if [ $# -ge 1 ]; then
        UPDATE_INFORMATION="-u zsync|$1"
fi
./appimagetool --sign --comp gzip $UPDATE_INFORMATION $APPDIR $APPNAME
)

echo $APPNAME

