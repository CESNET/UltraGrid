# shellcheck shell=sh
get_loader() {
        loaders='/lib64/ld-linux-*so* /lib/ld-linux-*so* /lib*/ld-linux-*so*'
        for n in $loaders; do
                for m in $n; do
                        if [ -x "$m" ]; then
                                echo "$m"
                                return
                        fi
                done
        done
}

# @param $1 UG library name
# @param $2 system preloaded library pattern
#
set_ld_preload() {
        ug_module_lib=$AI_LIB_PATH/ultragrid/$1
        if [ ! -f "$ug_module_lib" ]; then
                return
        fi
        loader=$(get_loader)
        if [ ! -x "$loader" ]; then
                return
        fi
        S_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
        # shellcheck disable=SC2154 # assigned in AppRun (this file is sourced)
        LD_LIBRARY_PATH=$orig_ld_library_path
        system_lib=$(LD_TRACE_LOADED_OBJECTS=1 $loader "$ug_module_lib" | grep "$2" | grep -v 'not found' | awk '{print $3}')
        LD_LIBRARY_PATH=$S_LD_LIBRARY_PATH
        if [ -n "$system_lib" ]; then
                export LD_PRELOAD="$system_lib"${LD_PRELOAD:+":$LD_PRELOAD"}
        fi
}

