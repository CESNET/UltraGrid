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

$srcdir/configure --enable-gpl $@

cd $ORIGDIR

