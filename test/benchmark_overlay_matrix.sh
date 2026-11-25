#!/bin/bash
#
# Matrix benchmark: 3 scenarios x 3 resolutions x 3 frame rates
# (and 5 filters for the scaled scenario), with per-cell frame budget
# in milliseconds and an OVER marker when avg total exceeds budget.
#
# Scenarios:
#   static       — file loaded once, no hot-reload (fast path)
#   ball_native  — hot-reload at output res, no per-reload scale
#   ball_scaled  — hot-reload of small PAMs, per-reload libswscale
#
# Args (env): SECS=<seconds per cell, default 8>
#
# Author: Ben Roeder <ben@sohonet.com>
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob

set -e

UV="${UV:-./bin/uv}"
[ -x "$UV" ] || { echo "uv binary not found at $UV"; exit 2; }
UV=$(cd "$(dirname "$UV")" && pwd)/$(basename "$UV")
BASE=$(cd "$(dirname "$UV")/.." && pwd)

SECS="${SECS:-8}"
# DISPLAY=sdl shows each cell in an SDL window so you can watch it run.
# Default is "dummy" which is invisible but more representative of pure
# postprocess cost (no SDL render in the pipeline).
DISPLAY_KIND="${DISPLAY:-dummy}"
# PP_EXTRA is appended to every postprocess option string. Use it to
# probe knobs without editing the script (e.g. PP_EXTRA="scale_threads=4").
PP_EXTRA="${PP_EXTRA:-}"
SMALL_SEQ="$BASE/../testimages"
SCRATCH=$(mktemp -d -t ovbench.XXXXXX)
LOG=$SCRATCH/uv.log
trap 'pkill -f "bin/uv" 2>/dev/null; rm -rf "$SCRATCH"' EXIT

# 200x100 solid green static PAM for the no-reload case.
STATIC_PAM=$SCRATCH/static.pam
python3 - <<EOF
import struct
W, H = 200, 100
with open("$STATIC_PAM", "wb") as f:
    f.write(f'P7\nWIDTH {W}\nHEIGHT {H}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'.encode())
    for _ in range(W * H):
        f.write(struct.pack('BBBB', 0, 255, 0, 255))
EOF

# Animation drives a CURRENT file by copying a sequence directory's frames
# at ~30 Hz. Returns when SECS is up.
animate_loop() {
    local seq_dir=$1 current=$2
    local end_t=$(( $(date +%s) + SECS ))
    while [ $(date +%s) -lt $end_t ]; do
        for f in "$seq_dir"/frame_*.pam; do
            cp "$f" "$current"
            sleep 0.033
            [ $(date +%s) -ge $end_t ] && break
        done
    done
}

# Run uv against the supplied postprocess args, optionally driving the
# mtime-reload animation, capture the Overlay-stats line.
# Args:
#   $1 size  $2 fps  $3 pp_args  $4 anim_dir(optional, "" = static)  $5 watched_pam
run_uv_and_animate() {
    local size=$1 fps=$2 pp_args=$3 anim_dir=$4 current=$5
    rm -f "$LOG"

    local display
    if [ "$DISPLAY_KIND" = "sdl" ]; then
        display="sdl"
    else
        display="dummy:codec=UYVY"
    fi
    local args="$pp_args"
    [ -n "$PP_EXTRA" ] && args="$args:$PP_EXTRA"
    "$UV" -t "testcard:size=$size:fps=$fps:codec=UYVY:pattern=ebu_bars" \
          -d "$display" \
          --postprocess "$args" \
          > "$LOG" 2>&1 &
    local pid=$!
    sleep 0.5

    if [ -n "$anim_dir" ]; then
        animate_loop "$anim_dir" "$current"
    else
        sleep "$SECS"
    fi

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    grep "Overlay stats" "$LOG" | tail -1
}

# Print one row of the table. Highlights cells over budget with " OVER".
# Args: scenario  resolution  fps  filter  raw_stats_line
print_row() {
    local scenario=$1 res=$2 fps=$3 filter=$4 line=$5
    local frames blend total budget mark
    frames=$(echo "$line" | sed -nE 's/.*Frames:[[:space:]]*([0-9]+).*/\1/p')
    blend=$(echo  "$line" | sed -nE 's/.*Avg blend:[[:space:]]*([0-9.]+) ms.*/\1/p')
    total=$(echo  "$line" | sed -nE 's/.*Avg total:[[:space:]]*([0-9.]+) ms.*/\1/p')
    budget=$(awk "BEGIN { printf \"%.2f\", 1000.0 / $fps }")
    if [ -n "$total" ]; then
        mark=$(awk -v t="$total" -v b="$budget" \
            'BEGIN { print (t + 0 > b + 0) ? " OVER" : "" }')
    fi
    printf "%-12s | %-9s | %3s | %-13s | %-7s | %7s ms | %7s ms (budget %5s ms)%s\n" \
        "$scenario" "$res" "$fps" "${filter:--}" \
        "${frames:-?}" "${blend:-?}" "${total:-?}" "$budget" "$mark"
}

RESOLUTIONS=(${RESOLUTIONS:-1280x720 1920x1080 3840x2160})
FPS_VALUES=(${FPS_VALUES:-24 30 60})
FILTERS=(${FILTERS:-nearest fast_bilinear bilinear bicubic lanczos})

printf "Matrix bench: %d s/cell, %s display, UYVY testcard.\n\n" \
    "$SECS" "$DISPLAY_KIND"
printf "%-12s | %-9s | %3s | %-13s | %-7s | %-9s | %-29s\n" \
    "scenario" "resolution" "fps" "filter" "frames" "avg blend" "avg total"
echo "------------------------------------------------------------------------------------------------"

CURRENT=$SCRATCH/current.pam

# 1) Static logo
for res in "${RESOLUTIONS[@]}"; do
    for fps in "${FPS_VALUES[@]}"; do
        line=$(run_uv_and_animate "$res" "$fps" \
            "overlay:file=$STATIC_PAM:position=center:perf" "" "")
        print_row "static" "$res" "$fps" "" "$line"
    done
done
echo

# 2) Ball at native resolution (no scaling)
for res in "${RESOLUTIONS[@]}"; do
    seq_dir="$BASE/../testimages_${res}/testimages"
    [ -d "$seq_dir" ] || { echo "  missing $seq_dir — skipping native at $res"; continue; }
    cp "$seq_dir/frame_0000.pam" "$CURRENT"
    for fps in "${FPS_VALUES[@]}"; do
        line=$(run_uv_and_animate "$res" "$fps" \
            "overlay:file=$CURRENT:position=top_left:perf" "$seq_dir" "$CURRENT")
        print_row "ball_native" "$res" "$fps" "" "$line"
    done
done
echo

# 3) Ball scaled up (5 filters x 3 resolutions x 3 fps)
[ -d "$SMALL_SEQ" ] || { echo "missing small ball seq at $SMALL_SEQ"; exit 2; }
for res in "${RESOLUTIONS[@]}"; do
    for fps in "${FPS_VALUES[@]}"; do
        for filter in "${FILTERS[@]}"; do
            cp "$SMALL_SEQ/frame_0000.pam" "$CURRENT"
            line=$(run_uv_and_animate "$res" "$fps" \
                "overlay:file=$CURRENT:position=top_left:scale=$res:scale_filter=$filter:perf" \
                "$SMALL_SEQ" "$CURRENT")
            print_row "ball_scaled" "$res" "$fps" "$filter" "$line"
        done
        echo
    done
done
