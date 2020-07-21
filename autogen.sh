#! /bin/sh
set -e

command -v automake >/dev/null 2>&1 || { echo >&2 "Automake missing. Aborting."; exit 1; }
command -v autoconf >/dev/null 2>&1 || { echo >&2 "Autoconf missing. Aborting."; exit 1; }

[ -d m4 ] || mkdir m4

# find MSVC if CUDA is present but no cl in PATH, don't override --with-cuda-host-compiler if explicit
cuda_host_compiler_arg_present() {
        while expr $# \> 0 >/dev/null; do
                if expr "x$1" : x--with-cuda-host-compiler >/dev/null; then
                        echo yes
                fi
                shift
        done
        echo no
}
if [ $(uname -o) = "Msys" -a $(cuda_host_compiler_arg_present "$@") = no ]; then
        CUDA_PRESENT=$(command -v nvcc >/dev/null && echo yes || echo no)
        CL_PRESENT=$(command -v cl >/dev/null && echo yes || echo no)
        if [ $CUDA_PRESENT = yes -a $CL_PRESENT = no ]; then
                VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
                INSTALL_DIR=$("$VSWHERE" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
                VERSION_FILE="$INSTALL_DIR/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
                if [ -f "$VERSION_FILE" ]; then
                        VERSION=$(cat "$VERSION_FILE")
                        PATH=$PATH:$(cygpath "$INSTALL_DIR/VC/Tools/MSVC/$VERSION/bin/HostX64/x64")
                        PATH_TO_CL=$(command -v cl.exe)
                        set -- "$@" "--with-cuda-host-compiler=$PATH_TO_CL"
                fi
        fi
fi

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`

cd $srcdir
# install config.guess config.sub install-sh missing
RES=$(automake --add-missing -c 2>&1 || true) # actual call will fail - we do not have Makefile.am
if test -n "$RES" -a -z "$(echo $RES | grep Makefile.am)"; then
        echo "$RES"
        exit 1
fi
# Running autoreconf is preferred over aclocal/autoheader/autoconf.
# It, however, needs to be a little bit bent because we do not use automake.
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

$srcdir/configure $CONFIGURE_OPTS "$@"

cd $ORIGDIR

