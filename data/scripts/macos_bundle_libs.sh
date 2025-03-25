#!/bin/sh -eu

dylibbundler=$1
dylibbundler_flags=$2
bundle=$3

rpath=@executable_path/../libs/

for n in "$bundle"/Contents/MacOS/*; do
        # shellcheck disable=SC2086 # intentional, even $dylibbundler
        # can have flags like 'dylibbundler -f'; obvious for _flags
        echo quit | $dylibbundler $dylibbundler_flags -of -cd -b \
                -d "$bundle/Contents/libs/" -p "$rpath" -x "$n"
done

# Remove multiple LC_RPATH occurences of @executable_path/../libs/
# as can be seen by cmd `otool -l $lib | grep -B 1 -A 2 LC_RPATH`
# which seem to be refused by dyld since macOS 15.4 as can be seen
# here <https://github.com/CESNET/UltraGrid/issues/436>. The multiplied
# paths are caused by dylibbundler LC_RPATH replacements.

## output number of $rpath occurences in 'cmd LC_RPATH'
num_rel_lc_rpath() {
        otool -l "$1" | sed -n '/LC_RPATH/,+2p' | grep -c "$rpath" || true
}

find "$bundle" -type f -print0 |
        while IFS= read -r -d '' n; do
                count=$(num_rel_lc_rpath "$n")
                # remove all but at most one LC_RPATH $path occurences
                while [ "$count" -gt 1 ]; do
                        install_name_tool -delete_rpath "$rpath" "$n"
                        count=$((count - 1))
                done
        done
