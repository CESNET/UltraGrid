#! /bin/sh

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir

if [ `uname -s` = "Darwin" ]; then
        LIBTOOLIZE=glibtoolize
else 
        LIBTOOLIZE=libtoolize
fi

aclocal && \
autoheader && \
$LIBTOOLIZE --copy && \
autoconf && \
$srcdir/configure $@
STATUS=$?

cd $ORIGDIR

exit $STATUS

