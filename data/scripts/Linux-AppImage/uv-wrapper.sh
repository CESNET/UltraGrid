#!/bin/sh

set -u

get_loader() {
        LOADERS='/lib64/ld-linux-*so* /lib/ld-linux-*so* /lib*/ld-linux-*so*'
        for n in $LOADERS; do
                for m in `ls $n`; do
                        if [ -x $m ]; then
                                echo $m
                                return
                        fi
                done
        done
}

set_ld_preload() {
        if [ ! -f $DIR/lib/ultragrid/ultragrid_aplay_jack.so ]; then
                return
        fi
        local LOADER=$(get_loader)
        if [ ! -x "$LOADER" ]; then
                return
        fi
        S_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=
        JACK_LIB=$(LD_TRACE_LOADED_OBJECTS=1 $LOADER $DIR/lib/ultragrid/ultragrid_aplay_jack.so | grep libjack | grep -v 'not found' | awk '{print $3}')
        LD_LIBRARY_PATH=$S_LD_LIBRARY_PATH
        if [ -n "$JACK_LIB" ]; then
                export LD_PRELOAD=$JACK_LIB${LD_PRELOAD:+" $LD_PRELOAD"}
        fi
}

DIR=`dirname $0`
export LD_LIBRARY_PATH=$DIR/lib${LD_LIBRARY_PATH:+":$LD_LIBRARY_PATH"}
# there is an issue with running_from_path() which evaluates this executable
# as being system-installed
#export PATH=$DIR/bin:$PATH
set_ld_preload

exec $DIR/bin/uv "$@"
