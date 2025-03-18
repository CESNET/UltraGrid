#!/bin/sh -eu
#
## Remove multiple LC_RPATH occurences of @executable_path/../libs/
## as can be seen by cmd `otool -l $lib | grep -B 1 -A 2 LC_RPATH`
## which seem to be refused by dyld since macOS 15.4 as can be seen
## here <https://github.com/CESNET/UltraGrid/issues/436>. The multiplied
## paths are caused by dylibbundler LC_RPATH replacements.
##
## @param $1 bundle path

rpath=@executable_path/../libs/

# output number of $rpath occurences in 'cmd LC_RPATH'
num_rel_lc_rpath() {
        otool -l "$1" | sed -n '/LC_RPATH/,+2p' | grep -c "$rpath" || true
}

find "$1" -type f -print0 |
        while IFS= read -r -d '' n; do
                count=$(num_rel_lc_rpath "$n")
                # remove all but at most one LC_RPATH $path occurences
                while [ "$count" -gt 1 ]; do
                        install_name_tool -delete_rpath "$rpath" "$n"
                        count=$((count - 1))
                done
        done
