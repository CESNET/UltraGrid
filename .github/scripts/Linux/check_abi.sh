#!/bin/sh -eu

# Checks libc/ibstdc++ ABI version
# see also https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html

## @todo
## consider removing semver.sh and utilize sort -V to compare

GLIBC_MAX=$1
GLIBCXX_MAX=$2
CXX_MAX=$3
shift 3

SEMVER_CMP=$(dirname $0)/utils/semver.sh

if [ ! -x $SEMVER_CMP ]; then
        echo "semver.sh script not present!" >&2
        exit 2
fi

while test $# -gt 0; do
        if [ ! -f $1 ]; then
                shift
                continue
        fi
        GLIBC_CUR=$(ldd -r -v $1 | sed -n 's/.*(GLIBC_\([0-9.]*\)).*/\1/p'  | sort -V | tail -n 1)
        ## @todo
        ## perpaps use ldd as well for the remaining 2?
        GLIBCXX_CUR=$(nm $1 | sed -n 's/.*GLIBCXX_\([0-9.]*\).*/\1/p' | sort -V | tail -n 1)
        CXX_CUR=$(nm $1 | sed -n 's/.*CXXABI_\([0-9.]*\).*/\1/p' | sort -V | tail -n 1)
        if [ -n "$GLIBC_CUR" -a "$($SEMVER_CMP $GLIBC_CUR $GLIBC_MAX)" -gt 0 ]; then
                echo "$1: GLIBC $GLIBC_CUR ($GLIBC_MAX required)" 1>&2
                exit 1
        fi
        if [ -n "$GLIBCXX_CUR" -a "$($SEMVER_CMP $GLIBCXX_CUR $GLIBCXX_MAX)" -gt 0 ]; then
                echo "$1: GLIBCXX $GLIBCXX_CUR ($GLIBCXX_MAX required)" 1>&2
                exit 1
        fi
        if [ -n "$CXX_CUR" -a "$($SEMVER_CMP $CXX_CUR $CXX_MAX)" -gt 0 ]; then
                echo "$1: CXX $CXX_CUR ($CXX_MAX required)" 1>&2
                exit 1
        fi
        shift
done

