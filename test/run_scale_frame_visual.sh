#!/bin/bash
#
# Visual demo of scale=frame.
#
# Builds a 480x270 PAM with eight vertical colour bars
# (red/green/blue/white × 2) and shows it through SDL twice: first
# WITHOUT scale=frame (so the user sees the PAM at native 480x270 —
# a clearly-recognisable badge in the centre, a quarter of a 1080p
# frame), then WITH scale=frame (the same PAM stretched across the
# full window). The eight colour bars make the horizontal stretch
# unambiguous: the bar boundaries land at frame_w * k/8 for each k.
#
# Then it walks several resolutions back-to-back to show that
# scale=frame produces a correctly-sized overlay at each, without
# the user having to recompute scale=WxH.
#
# Args: [seconds_per_window]   default 4
#
# Author: Ben Roeder <ben@sohonet.com>
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob

set -e

UV="${UV:-./bin/uv}"
[ -x "$UV" ] || { echo "uv binary not found at $UV"; exit 2; }
UV=$(cd "$(dirname "$UV")" && pwd)/$(basename "$UV")

SECS="${1:-4}"
RESOLUTIONS=("1280x720@30" "1920x1080@30" "3840x2160@30")

PAM=/tmp/overlay_scale_frame_visual.pam
LOG=/tmp/overlay_scale_frame_visual.log

# 480x270 PAM (1/4 of a 1080p frame, ~6% of area at 4K). Eight vertical
# colour bars: red, green, blue, white repeated twice. At native size
# you see a small recognisable badge in the centre; with scale=frame
# the same eight bars cover the full window so the stretch is obvious.
python3 - <<EOF
import struct
W, H = 480, 270
COLS = [
    (0xFF, 0x00, 0x00),  # red
    (0x00, 0xFF, 0x00),  # green
    (0x00, 0x00, 0xFF),  # blue
    (0xFF, 0xFF, 0xFF),  # white
    (0xFF, 0x00, 0x00),
    (0x00, 0xFF, 0x00),
    (0x00, 0x00, 0xFF),
    (0xFF, 0xFF, 0xFF),
]
COL_W = W // len(COLS)   # 60 px per bar
with open("$PAM", "wb") as f:
    f.write(f'P7\nWIDTH {W}\nHEIGHT {H}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'.encode())
    for _ in range(H):
        for c in range(W):
            r, g, b = COLS[c // COL_W]
            f.write(struct.pack('BBBB', r, g, b, 0xFF))
EOF

cleanup() {
    pkill -f "bin/uv" 2>/dev/null || true
    rm -f "$PAM" "$LOG"
}
trap cleanup INT TERM EXIT

run_one() {
    local label=$1 size=$2 fps=$3 pp=$4
    echo
    echo "====================================================================="
    echo "  $label   $size@${fps}fps   (${SECS}s)"
    echo "====================================================================="
    "$UV" -t "testcard:size=$size:fps=$fps:codec=RGBA:pattern=ebu_bars" \
          -d sdl \
          --postprocess "$pp" \
          > "$LOG" 2>&1 &
    local pid=$!
    sleep "$SECS"
    kill "$pid" 2>/dev/null || true
    pkill -f "bin/uv" 2>/dev/null || true
    wait 2>/dev/null || true
    grep -iE "(loaded|overlay stats|unsupported|fail)" "$LOG" \
        | sed 's/^/  /' | head -6 || true
}

# Comparison at 1080p: native vs scale=frame.
echo
echo "#####################################################################"
echo "#  Comparison: same 480x270 PAM at 1920x1080"
echo "#####################################################################"
run_one "Native 480x270 (no scale)" 1920x1080 30 \
        "overlay:file=$PAM:position=center"
run_one "scale=frame (stretched)" 1920x1080 30 \
        "overlay:file=$PAM:scale=frame:scale_filter=nearest:perf"

# Walk resolutions with scale=frame: each window should show the
# overlay covering the full frame regardless of source resolution.
echo
echo "#####################################################################"
echo "#  scale=frame across multiple source resolutions"
echo "#####################################################################"
for res in "${RESOLUTIONS[@]}"; do
    size=${res%@*}
    fps=${res#*@}
    run_one "scale=frame" "$size" "$fps" \
            "overlay:file=$PAM:scale=frame:scale_filter=nearest:perf"
done

echo
echo "Done — walked native and scale=frame at ${#RESOLUTIONS[@]} resolutions."
