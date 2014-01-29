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

$srcdir/configure --enable-gpl $@

cd $ORIGDIR

