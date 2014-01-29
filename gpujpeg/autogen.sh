#!/bin/sh
srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

if [ `uname -s` = "Darwin" ]; then
    LIBTOOLIZE=glibtoolize
else 
    LIBTOOLIZE=libtoolize
fi

autoheader && \
$LIBTOOLIZE --copy && \
( [ -d m4 ] || mkdir m4 ) && \
aclocal -I m4 && \
automake --copy --add-missing && \
autoconf && \
./configure "$@"

STATUS=$?

cd $ORIGDIR

([ $STATUS -eq 0 ] && echo "Autogen done." ) || echo "Autogen failed."

exit $STATUS

