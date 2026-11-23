#!/bin/bash
#
# Visual demo: launches SDL with the overlay across each supported codec
# in turn so a human can confirm the overlay actually appears correctly.
#
# Args: [seconds_per_codec]   default 6
#
# Author: Ben Roeder <ben@sohonet.com>
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob

set -e

UV="${UV:-./bin/uv}"
[ -x "$UV" ] || { echo "uv binary not found at $UV"; exit 2; }
UV=$(cd "$(dirname "$UV")" && pwd)/$(basename "$UV")

SECS="${1:-6}"
# Resolutions to walk. Format: WxH@FPS. Override via command-line args 2+.
if [ $# -gt 1 ]; then
    shift
    RESOLUTIONS=("$@")
else
    RESOLUTIONS=("1920x1080@60" "3840x2160@30")
fi
PAM=/tmp/overlay_visual.pam

# 200x100 solid green PAM. Soft edge is applied by the postprocessor.
python3 - <<EOF
import struct
W, H = 200, 100
with open("$PAM", "wb") as f:
    f.write(f'P7\nWIDTH {W}\nHEIGHT {H}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'.encode())
    for _ in range(W * H):
        f.write(struct.pack('BBBB', 0, 255, 0, 255))
EOF

# Codecs supported by overlay's blend dispatch. SDL accepts the ones in
# its preferred set; for codecs SDL doesn't accept directly we let the
# pipeline negotiate (the postprocess still runs against whatever format
# arrives at it).
CODECS=(RGBA RGB UYVY YUYV v210 R10k R12L Y416 I420)

trap 'pkill -f "bin/uv" 2>/dev/null; rm -f "$PAM" "$PAM.log"; exit' INT TERM

for res in "${RESOLUTIONS[@]}"; do
    size=${res%@*}
    fps=${res#*@}
    echo
    echo "####################################################################"
    echo "#  Resolution $size @ ${fps}fps"
    echo "####################################################################"

    for cs in "${CODECS[@]}"; do
        echo
        echo "===================================================================="
        echo "  $cs   $size@${fps}   (showing for ${SECS}s)"
        echo "===================================================================="
        "$UV" -t "testcard:size=$size:fps=$fps:codec=$cs:pattern=ebu_bars" \
              -d sdl \
              --postprocess "overlay:file=$PAM:position=top_right:custom_x=-30:custom_y=30:soft_edge=16:perf" \
              > "$PAM.log" 2>&1 &
        pid=$!
        sleep "$SECS"
        kill "$pid" 2>/dev/null || true
        pkill -f "bin/uv" 2>/dev/null || true
        wait 2>/dev/null || true
        grep -iE "(loaded|reconf|stats|unsupported|error|fail)" "$PAM.log" \
            | sed 's/^/  /' | head -8 || true
    done
done

rm -f "$PAM" "$PAM.log"
echo
echo "Done — walked through ${#CODECS[@]} codecs at ${#RESOLUTIONS[@]} resolutions."
