#!/bin/sh -eu
## Install speex and speexdsp
##
## Normally try to use git submodules present in the repository. If not possible
## it tries to either clone (.git directory stripped) or to download release
## (if git not found).

if command -v pkg-config >/dev/null && pkg-config speexdsp; then
        echo "Using system SpeexDSP"
        exit 0
fi

. $(dirname $0)/fetch_submodule.sh

if fetch_submodule speexdsp http://downloads.us.xiph.org/releases/speex/speexdsp-1.2.0.tar.gz https://gitlab.xiph.org/xiph/speexdsp 1
then
        SUBMODULE_UPDATED=no
else
        SUBMODULE_UPDATED=yes
fi

printf "Configuring speexdsp... "
if [ -f ext-deps/speexdsp/include/speex/speexdsp_config_types.h -a $SUBMODULE_UPDATED = no ]; then
        echo "not needed"
else
        cd ext-deps/speexdsp
        ./autogen.sh
        ./configure
        cd ../..
fi
