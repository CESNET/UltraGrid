#! /bin/sh
set -e

command -v automake >/dev/null 2>&1 || { echo >&2 "Automake missing. Aborting."; exit 1; }
command -v autoconf >/dev/null 2>&1 || { echo >&2 "Autoconf missing. Aborting."; exit 1; }

[ -d m4 ] || mkdir m4

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.
ORIGDIR=`pwd`
cd $srcdir

. ./data/scripts/add_cl_if_not_present.sh # add --with-cuda-host-compiler=<cl> to current params (Win)

# install config.guess config.sub install-sh missing
echo "Running automake..."
RES=$(automake --add-missing -c 2>&1 || true) # actual call will fail - we do not have Makefile.am
if test -n "$RES" -a -z "$(echo $RES | grep Makefile.am)"; then
        echo "$RES"
        exit 1
fi
# Running autoreconf is preferred over aclocal/autoheader/autoconf.
# It, however, needs to be a little bit bent because we do not use automake.
echo "Running autoreconf..."
RES=$(autoreconf -i 2>&1 || true)
# check if the error was the expected absence of Makefile.am or something else - then fail
if test -n "$RES" -a -z "$(echo $RES | grep Makefile.am)"; then
        echo "$RES"
        exit 1
fi

cd $ORIGDIR

echo "Running configure..."
"$srcdir"/configure "$@"

