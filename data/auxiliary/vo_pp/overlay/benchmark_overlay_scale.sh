#!/bin/bash
#
# Benchmark the upscale path: 5 filters x 3 resolutions, mtime-driven
# at 30 fps reload (the bouncing ball animation), captures the
# Overlay-stats line per run and prints a summary.
#
# Author: Ben Roeder <ben@sohonet.com>
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob

set -e

UV="${UV:-./bin/uv}"
[ -x "$UV" ] || { echo "uv binary not found at $UV"; exit 2; }
UV=$(cd "$(dirname "$UV")" && pwd)/$(basename "$UV")

SECS="${SECS:-12}"
SEQ_DIR="${SEQ_DIR:-../testimages}"
SEQ_DIR=$(cd "$SEQ_DIR" 2>/dev/null && pwd) \
    || { echo "sequence dir $SEQ_DIR not found (run generate_bouncing_ball.py first)"; exit 2; }

CURRENT=$(mktemp -t bench-current.pam.XXXXXX)
LOG=$(mktemp -t bench-uv.log.XXXXXX)
trap 'pkill -f "bin/uv" 2>/dev/null; rm -f "$CURRENT" "$LOG"' EXIT

cp "$SEQ_DIR/frame_0000.pam" "$CURRENT"

FILTERS=(nearest fast_bilinear bilinear bicubic lanczos)
RESOLUTIONS=("1280x720@30" "1920x1080@60" "3840x2160@30")

# Drives the mtime-reload loop from the foreground while uv runs in the
# background; logs go to $LOG so we can extract the stats line at the end.
run_case() {
    local size=$1 fps=$2 filter=$3
    rm -f "$LOG"
    cp "$SEQ_DIR/frame_0000.pam" "$CURRENT"

    "$UV" -t "testcard:size=$size:fps=$fps:codec=UYVY:pattern=ebu_bars" \
          -d "dummy:codec=UYVY" \
          --postprocess "overlay:file=$CURRENT:position=top_left:scale=$size:scale_filter=$filter:perf" \
          > "$LOG" 2>&1 &
    local pid=$!
    sleep 0.5

    # Animate at ~30 fps for $SECS seconds (3 loops of 120 frames = 12s).
    local end_t=$(( $(date +%s) + SECS ))
    while [ $(date +%s) -lt $end_t ]; do
        for f in "$SEQ_DIR"/frame_*.pam; do
            cp "$f" "$CURRENT"
            sleep 0.033
            [ $(date +%s) -ge $end_t ] && break
        done
    done

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    grep "Overlay stats" "$LOG" | tail -1
}

printf "Bench: %d s per cell, dummy display, UYVY, mtime-driven 30 fps reload\n\n" "$SECS"
printf "%-18s | %-13s | %-7s | %-9s | %-9s\n" \
    "Resolution" "Filter" "Frames" "Avg blend" "Avg total"
echo "---------------------------------------------------------------------"
for res in "${RESOLUTIONS[@]}"; do
    size=${res%@*}
    fps=${res#*@}
    for filter in "${FILTERS[@]}"; do
        line=$(run_case "$size" "$fps" "$filter")
        # Extract Frames / Avg blend / Avg total from the stats line.
        frames=$(echo "$line"  | sed -nE 's/.*Frames:[[:space:]]*([0-9]+).*/\1/p')
        blend=$(echo "$line"   | sed -nE 's/.*Avg blend:[[:space:]]*([0-9.]+) ms.*/\1/p')
        total=$(echo "$line"   | sed -nE 's/.*Avg total:[[:space:]]*([0-9.]+) ms.*/\1/p')
        printf "%-18s | %-13s | %-7s | %7s ms | %7s ms\n" \
            "$size@$fps" "$filter" "${frames:-?}" "${blend:-?}" "${total:-?}"
    done
    echo
done
