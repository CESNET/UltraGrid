#!/bin/sh -eu
## Install speex and speexdsp
##
## Normally try to use git submodules present in the repository. If not possible
## it tries to either clone (.git directory stripped) or to download release
## (if git not found).

. $(dirname $0)/fetch_submodule.sh

for module in speex speexdsp; do
        if command -v pkg-config >/dev/null && pkg-config $module; then
                echo "Using system $module"
                continue
        fi

        if fetch_submodule $module http://downloads.us.xiph.org/releases/speex/${module}-1.2.0.tar.gz https://gitlab.xiph.org/xiph/$module 1
        then
                SUBMODULE_UPDATED=no
        else
                SUBMODULE_UPDATED=yes
        fi

        printf "Configuring ${module}... "
        if [ -f ext-deps/$module/include/speex/${module}_config_types.h -a $SUBMODULE_UPDATED = no ]; then
                echo "not needed"
        else
                cd ext-deps/$module
                ./autogen.sh
                ./configure
                cd ../..
        fi
done

