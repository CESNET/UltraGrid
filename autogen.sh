#! /bin/sh
set -e

[ -d m4 ] || mkdir m4

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

cd $srcdir
aclocal
autoheader
autoconf

CONFIGURE_OPTS="--enable-gpl"

if [ -n "$DEBUG" ]; then
        CONFIGURE_OPTS="$CONFIGURE_OPTS --enable-debug"
fi

cd $ORIGDIR

$srcdir/configure $CONFIGURE_OPTS $@

cd $ORIGDIR

