#!/bin/sh -eux

APPDIR=UltraGrid.AppDir
ARCH=`uname -m`
DATE=`date +%Y%m%d`
GLIBC_VERSION=`ldd --version | head -n 1 | sed 's/.*\ \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/'`
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

# glibc libraries should not be bundled
# Taken from https://gitlab.com/probono/platformissues
# libnsl.so.1 is not removed - is not in Fedora 28 by default
for n in ld-linux.so.2 ld-linux-x86-64.so.2 libanl.so.1 libBrokenLocale.so.1 libcidn.so.1 lib crypt.so.1 libc.so.6 libdl.so.2 libm.so.6 libmvec.so.1 libnss_compat.so.2 libnss_db.so.2 libnss_dns.so.2 libnss_files.so.2 libnss_hesiod.so.2 libnss_nisplus.so.2 libnss_nis.so.2 libpthread.so.0 libresolv.so.2 librt.so.1 libthread_db.so.1 libutil.so.1; do
        if [ -f $APPDIR/lib/$n ]; then
                rm $APPDIR/lib/$n
        fi
done

# consult https://github.com/probonopd/linuxdeployqt/blob/master/tools/linuxdeployqt/excludelist.h
( cd $APPDIR/lib; rm -f libasound.so.2 libdrm.so.2 libEGL.so.1 libGL.so.1 libGLdispatch.so.0 libstdc++.so.6  libX11-xcb.so.1 libX11.so.6 libXau.so.6 libXcursor.so.1 libXdmcp.so.6 libXext.so.6 li bXfixes.so.3 libXi.so.6 libXinerama.so.1 libXrandr.so.2 libXrender.so.1 libXtst.so.6 libXxf86vm.so.1 libxcb* libxshm* )
( cd $APPDIR/lib; rm -f libcmpto* ) # remove non-free components

cp data/scripts/Linux-AppImage/AppRun data/scripts/Linux-AppImage/uv-wrapper.sh data/ultragrid.png $APPDIR
cp data/uv-qt.desktop $APPDIR/ultragrid.desktop
wget --no-verbose https://github.com/AppImage/AppImageUpdate/releases/download/continuous/appimageupdatetool-x86_64.AppImage -O $APPDIR/appimageupdatetool
chmod ugo+x $APPDIR/appimageupdatetool

if [ -n "$appimage_key" ]; then
        echo "$appimage_key" >> key
        gpg --import key
fi

wget --no-verbose https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -O appimagetool && chmod 755 appimagetool
./appimagetool --sign --comp gzip -u "zsync|https://github.com/${GITHUB_REPOSITORY-CESNET/UltraGrid}/releases/download/nightly/UltraGrid-nightly-x86_64.AppImage.zsync" $APPDIR $APPNAME
)

echo $APPNAME

