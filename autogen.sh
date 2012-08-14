#! /bin/sh
set -e

# variables
if [ `uname -s` = "Darwin" ]; then
        LIBTOOLIZE=glibtoolize
else
        LIBTOOLIZE=libtoolize
fi


srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

cd $srcdir
aclocal
autoheader
$LIBTOOLIZE --copy
autoconf

cd $srcdir/gpujpeg
DO_NOT_CONFIGURE="1" ./autogen.sh
cd -

[ -n "$DO_NOT_CONFIGURE" ] || $srcdir/configure $@

cd $ORIGDIR

