#!/bin/sh
srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

if [ `uname -s` = "Darwin" ]; then
    LIBTOOLIZE=glibtoolize
else 
    LIBTOOLIZE=libtoolize
fi

if [ ! -x ../ltmain.sh ]
then
        cd ..
        $LIBTOOLIZE --copy
        cd -
fi

autoheader && \
$LIBTOOLIZE --copy && \
( [ -d m4 ] || mkdir m4 ) && \
aclocal -I m4 && \
automake --copy --add-missing && \
autoconf && \
[ -n "$DO_NOT_CONFIGURE" ] || ./configure "$@"

STATUS=$?

cd $ORIGDIR

([ $STATUS -eq 0 ] && echo "Autogen done." ) || echo "Autogen failed."

exit $STATUS

