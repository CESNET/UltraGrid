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
./data/scripts/install_speexdsp.sh
. ./data/scripts/fetch_submodule.sh
fetch_submodule zfec https://files.pythonhosted.org/packages/1c/bf/b87a31205fcd2e0e4b4c9a3f7bf6f5a231e199bec5f654d7c5ac6fcec349/zfec-1.5.5.tar.gz https://github.com/tahoe-lafs/zfec

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

CONFIGURE_OPTS=

if [ -n "$DEBUG" ]; then
        CONFIGURE_OPTS="$CONFIGURE_OPTS --enable-debug"
fi

cd $ORIGDIR

echo "Running configure..."
$srcdir/configure $CONFIGURE_OPTS "$@"

cd $ORIGDIR

