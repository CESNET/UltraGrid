#! /bin/sh
set -e

[ -d m4 ] || mkdir m4

# variables
if [ `uname -s` = "Darwin" ]; then
        LIBTOOLIZE=glibtoolize
else if [ `uname -s` = "Linux" ]; then
        LIBTOOLIZE=libtoolize
else # Windows
        LIBTOOLIZE=true
fi
fi


srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

cd $srcdir
aclocal
autoheader
$LIBTOOLIZE --copy
autoconf

# configure JPEG only on Linux/OSX
if ! `uname -s | grep -q '^MINGW32'`; then
        cd $srcdir/gpujpeg
        DO_NOT_CONFIGURE="1" ./autogen.sh
        cd -
fi

[ -n "$DO_NOT_CONFIGURE" ] || $srcdir/configure --enable-gpl $@

cd $ORIGDIR

