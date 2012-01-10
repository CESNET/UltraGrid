#! /bin/sh

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir

aclocal && \
autoheader && \
autoconf && \
$srcdir/configure $@
STATUS=$?

cd $ORIGDIR

exit $STATUS

