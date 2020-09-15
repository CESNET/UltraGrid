#!/bin/sh -eu

# Checks libstdc++ ABI version
# see also https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html

GLIBCXX_MAX=$1
CXX_MAX=$2
shift 2

SEMVER_CMP=$(dirname $0)/utils/semver.sh

if [ ! -x $SEMVER_CMP ]; then
        echo "semver.sh script not present!" >&2
        exit 2
fi

while test $# -gt 0; do
        if [ -f $1 ]; then
                shift
                continue
        fi
        GLIBCXX_CUR=$(nm $1 | sed -n 's/.*GLIBCXX_\([0-9.]*\).*/\1/p' | sort -V | tail -n 1)
        CXX_CUR=$(nm $1 | sed -n 's/.*CXXABI_\([0-9.]*\).*/\1/p' | sort -V | tail -n 1)
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

