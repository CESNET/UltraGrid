#! /bin/sh
set -e

[ -d m4 ] || mkdir m4

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

cd $srcdir
if test -d .git ; then
        git submodule update --init ldgm
else
        git clone "http://seth.ics.muni.cz/git/ldgm.git"
fi
aclocal
autoheader
autoconf

CONFIGURE_OPTS=

if [ -n "$DEBUG" ]; then
        CONFIGURE_OPTS="$CONFIGURE_OPTS --enable-debug"
fi

cd $ORIGDIR

$srcdir/configure $CONFIGURE_OPTS $@

cd $ORIGDIR

