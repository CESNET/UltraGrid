#!/bin/sh -eux

export CPATH=/usr/local/include${CPATH:+":$CPATH"}
EXTRA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib
export LIBRARY_PATH=$EXTRA_LIB_PATH${LIBRARY_PATH:+":$LIBRARY_PATH"}
export LD_LIBRARY_PATH=$EXTRA_LIB_PATH${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
export PATH="/usr/local/bin:$PATH"
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:+":$PKG_CONFIG_PATH"}

APPDIR=UltraGrid.AppDir
ARCH=$(dpkg --print-architecture)
APPNAME=UltraGrid-latest-${ARCH}.AppImage

./autogen.sh --enable-plugins

mkdir $APPDIR
mkdir tmpinstall
make DESTDIR=tmpinstall install
mv tmpinstall/usr/local/* $APPDIR

for n in "$APPDIR"/bin/* "$APPDIR"/lib/ultragrid/*
do
	for lib in $(ldd "$n" | awk '{ print $3 }'); do [ ! -f "$lib" ] || cp "$lib" $APPDIR/lib; done
done

# glibc libraries should not be bundled
# Taken from https://gitlab.com/probono/platformissues
# libnsl.so.1 is not removed - is not in Fedora 28 by default
for n in ld-linux.so.2 ld-linux-aarch64.so.1 ld-linux-armhf.so.3 libanl.so.1 libBrokenLocale.so.1 libcidn.so.1 libcrypt.so.1 libc.so.6 libdl.so.2 libm.so.6 libmvec.so.1 libnss_compat.so.2 libnss_db.so.2 libnss_dns.so.2 libnss_files.so.2 libnss_hesiod.so.2 libnss_nisplus.so.2 libnss_nis.so.2 libpthread.so.0 libresolv.so.2 librt.so.1 libthread_db.so.1 libutil.so.1
do
        if [ -f $APPDIR/lib/$n ]; then
                rm $APPDIR/lib/$n
        fi
done

( cd $APPDIR/lib; rm -f libasound.so.2 libdrm.so.2 libEGL.so.1 libGL.so.1 libGLdispatch.so.0 libstdc++.so.6  libX11-xcb.so.1 libX11.so.6 libXau.so.6 libXcursor.so.1 libXdmcp.so.6 libXext.so.6 libXfixes.so.3 libXi.so.6 libXinerama.so.1 libXrandr.so.2 libXrender.so.1 libXtst.so.6 libXxf86vm.so.1 libxcb* libxshm* )

cat << 'EOF' >> $APPDIR/uv-wrapper.sh
#!/bin/sh

set -u

DIR=`dirname $0`
export LD_LIBRARY_PATH=$DIR/lib${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
# there is an issue with running_from_path() which evaluates this executable
# as being system-installed
#export PATH=$DIR/bin:$PATH

exec $DIR/bin/uv "$@"
EOF

chmod 755 $APPDIR/uv-wrapper.sh

cat << 'EOF' >> $APPDIR/AppRun
#!/bin/sh

set -u

DIR=`dirname $0`
export LD_LIBRARY_PATH=$DIR/lib${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
# there is an issue with running_from_path() which evaluates this executable
# as being system-installed
#export PATH=$DIR/bin:$PATH
export QT_QPA_FONTDIR=$DIR/lib/fonts

usage() {
	printf "usage:\n"
	printf "\tUltraGrid --help\n"
	printf "\t\tprints this help\n"
	printf "\n"
	printf "\tUltraGrid --appimage-help\n"
	printf "\t\tprints AppImage related options\n"
	printf "\n"
	printf "\tUltraGrid --tool uv --help\n"
	printf "\t\tprints command-line UltraGrid help\n"
	printf "\n"
	printf "\tUltraGrid --tool <t> [args]\n"
	printf "\t\tinvokes specified tool\n"
	printf "\t\ttool may be: $(ls $DIR/bin | tr '\n' ' ')\n"
	printf "\n"
	printf "\tUltraGrid args\n"
	printf "\t\tinvokes command-line UltraGrid\n"
	printf "\n"
}

if [ $# -eq 0 ]; then
	usage
elif [ x"$1" = x"--tool" ]; then
	TOOL=$2
	shift 2
	$DIR/bin/$TOOL "$@"
elif [ x"$1" = x"-h" -o x"$1" = x"--help" ]; then
	usage
	exit 0
else
	$DIR/bin/uv "$@"
fi

exit $?
EOF
chmod 755 $APPDIR/AppRun

cp data/ultragrid.png $APPDIR/ultragrid.png
cp data/uv-qt.desktop $APPDIR/ultragrid.desktop

mkappimage --comp gzip $APPDIR # --sign
mv -- *.AppImage "$APPNAME"

