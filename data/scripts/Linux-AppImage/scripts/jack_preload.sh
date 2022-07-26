get_loader() {
        loaders='/lib64/ld-linux-*so* /lib/ld-linux-*so* /lib*/ld-linux-*so*'
        for n in $loaders; do
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
        loader=$(get_loader)
        if [ ! -x "$loader" ]; then
                return
        fi
        S_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=
        jack_lib=$(LD_TRACE_LOADED_OBJECTS=1 $loader $DIR/lib/ultragrid/ultragrid_aplay_portaudio.so | grep libjack | grep -v 'not found' | awk '{print $3}')
        LD_LIBRARY_PATH=$S_LD_LIBRARY_PATH
        if [ -n "$jack_lib" ]; then
                export LD_PRELOAD=$jack_lib${LD_PRELOAD:+" $LD_PRELOAD"}
        fi
}

set_ld_preload

