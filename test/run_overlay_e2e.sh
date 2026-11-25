#!/bin/bash
#
# End-to-end pixel verification for the overlay postprocessor.
#
# For each supported codec we run:
#   testcard (solid colour) -> overlay (top_left) -> dummy display dump
# then read back the dumped raw frame and assert the upper-left pixels
# match the overlay colour while the area outside the overlay matches
# the testcard background. This proves the whole pipeline (PAM load,
# positioning, format dispatch, plane offsets) — not just that frames
# flow.
#
# Author: Ben Roeder <ben@sohonet.com>
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob

set -e

UV="${UV:-./bin/uv}"
[ -x "$UV" ] || { echo "uv binary not found at $UV"; exit 2; }
# Resolve to absolute path so the cd-into-workdir below still finds it.
UV=$(cd "$(dirname "$UV")" && pwd)/$(basename "$UV")

WORK=$(mktemp -d -t ug-overlay-e2e.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

PAM=$WORK/overlay.pam
DUMP=$WORK/dummy
PASS=0
FAIL=0

# 24x2 solid green PAM. Width chosen so it survives block-pixel snapping
# for every codec we test: multiple of 6 (v210), 8 (R12L), 4, 2, 1.
OVERLAY_W=24
OVERLAY_H=2
{
    printf 'P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\n' \
        "$OVERLAY_W" "$OVERLAY_H"
    printf 'TUPLTYPE RGB_ALPHA\nENDHDR\n'
    for _ in $(seq 1 $((OVERLAY_W * OVERLAY_H))); do
        printf '\x00\xff\x00\xff'
    done
} > "$PAM"

FRAME_W=48
FRAME_H=4

# Run uv once and pull a single frame to $DUMP.
# Args: codec [extra_display_opts]
run_uv() {
    local codec=$1
    local display="dummy:codec=$codec:dump=oneshot:raw"
    rm -f "$DUMP".*
    cd "$WORK"
    "$UV" -t "testcard:size=${FRAME_W}x${FRAME_H}:fps=1:codec=$codec:pattern=blank=0xFF" \
          -d "$display" \
          --postprocess "overlay:file=$PAM:position=top_left" \
          >/dev/null 2>&1 &
    local pid=$!
    local n=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 0.2
        n=$((n + 1))
        if [ $n -gt 25 ]; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null || true
    cd - >/dev/null
}

# Dump the byte at offset $1 of the dump file as 2-digit hex.
hex_at() {
    od -An -tx1 -N1 -j"$1" "$DUMP".* | tr -d ' \n'
}

# Dump $2 bytes starting at offset $1 as a contiguous lowercase hex string.
hex_range() {
    od -An -tx1 -N"$2" -j"$1" "$DUMP".* | tr -d ' \n'
}

assert_eq() {
    local label=$1 expected=$2 actual=$3
    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label expected=$expected actual=$actual"
        FAIL=$((FAIL + 1))
    fi
}

# Pick any dump file produced by uv (the extension varies by codec).
dump_file() {
    ls "$DUMP".* 2>/dev/null | head -1
}

# Verify two byte ranges in the dump differ. With background = solid
# colour and overlay = different solid colour, these must always differ
# if the blend ran. We never decode codec-specific bytes here because
# the unit tests already prove the per-format math.
# Args: label overlay_off bg_off len
assert_differs() {
    local label=$1 overlay_off=$2 bg_off=$3 len=$4
    local f=$(dump_file)
    [ -n "$f" ] && [ -s "$f" ] || { echo "  FAIL: $label no dump file"; FAIL=$((FAIL+1)); return; }
    local o=$(od -An -tx1 -N"$len" -j"$overlay_off" "$f" | tr -d ' \n')
    local b=$(od -An -tx1 -N"$len" -j"$bg_off"      "$f" | tr -d ' \n')
    if [ "$o" != "$b" ]; then
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label overlay==bg at $overlay_off vs $bg_off ($o)"
        FAIL=$((FAIL + 1))
    fi
}

# RGBA / RGB get exact-byte assertions because the layout is trivial.
# Both codecs have no colour conversion — overlay green stays green.
test_rgba_exact() {
    echo "==> RGBA exact bytes"
    run_uv RGBA
    local f=$(dump_file)
    [ -n "$f" ] && [ -s "$f" ] || { echo "  no dump file"; FAIL=$((FAIL+1)); return; }
    assert_eq "RGBA pixel(0,0) is green" "00ff00ff" \
              "$(od -An -tx1 -N4 -j0 "$f" | tr -d ' \n')"
    # Pixel just past the overlay (column 24) is bg red.
    assert_eq "RGBA pixel(24,0) is red"  "ff0000ff" \
              "$(od -An -tx1 -N4 -j$((24 * 4)) "$f" | tr -d ' \n')"
}

test_rgb_exact() {
    echo "==> RGB exact bytes"
    run_uv RGB
    local f=$(dump_file)
    [ -n "$f" ] && [ -s "$f" ] || { echo "  no dump file"; FAIL=$((FAIL+1)); return; }
    assert_eq "RGB pixel(0,0) is green" "00ff00" \
              "$(od -An -tx1 -N3 -j0 "$f" | tr -d ' \n')"
    assert_eq "RGB pixel(24,0) is red"  "ff0000" \
              "$(od -An -tx1 -N3 -j$((24 * 3)) "$f" | tr -d ' \n')"
}

# Codecs where we just assert "overlay region differs from bg region".
# overlay_off / bg_off are both within row 0; the overlay's first sample
# vs the row's last sample (well past the overlay's right edge).
test_differs() {
    local codec=$1 overlay_off=$2 bg_off=$3 len=$4
    echo "==> $codec"
    run_uv "$codec"
    assert_differs "$codec overlay vs bg row0" "$overlay_off" "$bg_off" "$len"
}

test_rgba_exact
test_rgb_exact
test_differs UYVY  0 80 4    # 48-wide UYVY = 96 B/row; bg pair near the end
test_differs YUYV  0 80 4
test_differs v210  0 112 16   # 48-wide v210 = 128 B/row (aligned); bg at end
test_differs R10k  0 176 4    # 48-wide R10k = 192 B/row; bg near end
test_differs R12L  0 180 9    # 48-wide R12L = 6 groups * 36 = 216 B/row
test_differs Y416  0 320 8    # 48-wide Y416 = 384 B/row
# I420 Y plane is 48 B/row * 4 rows = 192 B; row 0 first vs row 0 last 4 px:
test_differs I420  0 44 4

# Threaded-blend parity: the same overlay through blend_threads=0 and
# blend_threads=4 must produce a byte-identical dump. Catches off-by-one
# mistakes in the row-stripe split.
test_blend_threads_parity() {
    echo "==> blend_threads parity (RGBA, 0 vs 4 workers)"
    rm -f "$DUMP".*
    cd "$WORK"
    "$UV" -t "testcard:size=64x16:fps=1:codec=RGBA:pattern=blank=0xFF" \
          -d "dummy:codec=RGBA:dump=oneshot:raw" \
          --postprocess "overlay:file=$PAM:position=top_left" \
          >/dev/null 2>&1 &
    local pid=$!
    local n=0
    while kill -0 "$pid" 2>/dev/null && [ $n -lt 25 ]; do sleep 0.2; n=$((n+1)); done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    cd - >/dev/null
    [ -s "$DUMP".rgba ] || { echo "  FAIL: serial dump missing or empty (uv crash?)"; FAIL=$((FAIL+1)); return; }
    cp "$DUMP".rgba "$WORK/serial.rgba"

    rm -f "$DUMP".*
    cd "$WORK"
    "$UV" -t "testcard:size=64x16:fps=1:codec=RGBA:pattern=blank=0xFF" \
          -d "dummy:dump=oneshot:raw" \
          --postprocess "overlay:file=$PAM:position=top_left:blend_threads=4" \
          >/dev/null 2>&1 &
    local pid2=$!
    local n=0
    while kill -0 "$pid2" 2>/dev/null && [ $n -lt 25 ]; do sleep 0.2; n=$((n+1)); done
    kill "$pid2" 2>/dev/null || true
    wait "$pid2" 2>/dev/null || true
    cd - >/dev/null

    # cmp -s passes for two empty files; guard against a uv crash in the
    # threaded run silently matching the serial dump.
    [ -s "$DUMP".rgba ] || { echo "  FAIL: threaded dump missing or empty (uv crash?)"; FAIL=$((FAIL+1)); return; }
    if cmp -s "$WORK/serial.rgba" "$DUMP".rgba; then
        PASS=$((PASS + 1))
    else
        echo "  FAIL: serial vs threaded output differs"
        FAIL=$((FAIL + 1))
    fi
}

test_blend_threads_parity

# Two overlays in the same -p invocation. The framework chains
# postprocessors with `,`; each runs in sequence with the previous
# stage's output as input. Verifies that two overlay instances coexist
# (independent state, no buffer aliasing) and that top_right positions
# flush against the right edge as documented.
test_dual_overlay_chain() {
    echo "==> dual overlay chain (RGBA, green top_left + red top_right)"
    local ddir=$WORK/dual
    mkdir -p "$ddir"
    # Green and red 24x2 PAMs (same dims as the main test PAM).
    local gpam=$ddir/green.pam
    local rpam=$ddir/red.pam
    for spec in "$gpam:00:ff:00" "$rpam:ff:00:00"; do
        local file=${spec%%:*}
        local rest=${spec#*:}
        local r=$(echo "$rest" | cut -d: -f1)
        local g=$(echo "$rest" | cut -d: -f2)
        local b=$(echo "$rest" | cut -d: -f3)
        {
            printf 'P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\n' \
                "$OVERLAY_W" "$OVERLAY_H"
            printf 'TUPLTYPE RGB_ALPHA\nENDHDR\n'
            for _ in $(seq 1 $((OVERLAY_W * OVERLAY_H))); do
                printf "\\x$r\\x$g\\x$b\\xff"
            done
        } > "$file"
    done

    cd "$ddir"
    "$UV" -t "testcard:size=${FRAME_W}x${FRAME_H}:fps=1:codec=RGBA:pattern=blank=0xFF" \
          -d "dummy:codec=RGBA:dump=oneshot:raw" \
          --postprocess "overlay:file=$gpam:position=top_left,overlay:file=$rpam:position=top_right" \
          >/dev/null 2>&1 &
    local pid=$!
    local n=0
    while kill -0 "$pid" 2>/dev/null && [ $n -lt 25 ]; do sleep 0.2; n=$((n+1)); done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    cd - >/dev/null

    local f=$(ls "$ddir"/dummy.* 2>/dev/null | head -1)
    [ -s "$f" ] || { echo "  FAIL: no dump"; FAIL=$((FAIL+1)); return; }

    # Pixel (0,0) — green overlay's first pixel (top_left).
    assert_eq "dual: pixel(0,0) is green" "00ff00ff" \
              "$(od -An -tx1 -N4 -j0 "$f" | tr -d ' \n')"
    # Pixel (FRAME_W - OVERLAY_W, 0) — red overlay's first pixel
    # (top_right places the overlay flush against the right edge).
    local red_off=$(( (FRAME_W - OVERLAY_W) * 4 ))
    assert_eq "dual: pixel(${FRAME_W}-${OVERLAY_W},0) is red" "ff0000ff" \
              "$(od -An -tx1 -N4 -j$red_off "$f" | tr -d ' \n')"
}

test_dual_overlay_chain

# Overlay larger than the frame, centred. Builds a 96x8 PAM split
# left-half=red / right-half=green; frame is 48x4. CENTER positioning
# slices the *middle* 48 columns out of the 96-wide overlay, which
# spans the red→green transition at overlay column 48. So the visible
# left half (frame x=0..23) shows red and the visible right half
# (x=24..47) shows green. Pre-fix this would have shown all-red
# because src_x defaulted to 0 and the blend read the overlay's
# leftmost slice instead of the centre.
test_oversized_center() {
    echo "==> oversized overlay (96x8) centred on 48x4 frame"
    local odir=$WORK/oversize
    mkdir -p "$odir"
    local pam=$odir/big.pam
    local OW=96 OH=8
    {
        printf 'P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\n' "$OW" "$OH"
        printf 'TUPLTYPE RGB_ALPHA\nENDHDR\n'
        for _ in $(seq 1 $OH); do
            for _ in $(seq 1 $((OW / 2))); do printf '\xff\x00\x00\xff'; done
            for _ in $(seq 1 $((OW / 2))); do printf '\x00\xff\x00\xff'; done
        done
    } > "$pam"

    cd "$odir"
    "$UV" -t "testcard:size=${FRAME_W}x${FRAME_H}:fps=1:codec=RGBA:pattern=blank=0xFF" \
          -d "dummy:codec=RGBA:dump=oneshot:raw" \
          --postprocess "overlay:file=$pam:position=center" \
          >/dev/null 2>&1 &
    local pid=$!
    local n=0
    while kill -0 "$pid" 2>/dev/null && [ $n -lt 25 ]; do sleep 0.2; n=$((n+1)); done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    cd - >/dev/null

    local f=$(ls "$odir"/dummy.* 2>/dev/null | head -1)
    [ -s "$f" ] || { echo "  FAIL: no dump"; FAIL=$((FAIL+1)); return; }

    # Visible 48 cols map to overlay cols 24..71. Cols 24..47 are red,
    # 48..71 are green. So frame pixel x=0 == overlay col 24 (red);
    # frame pixel x=24 == overlay col 48 (first green).
    assert_eq "oversized: pixel(0,0) is red"   "ff0000ff" \
              "$(od -An -tx1 -N4 -j0 "$f" | tr -d ' \n')"
    assert_eq "oversized: pixel(24,0) is green" "00ff00ff" \
              "$(od -An -tx1 -N4 -j$((24 * 4)) "$f" | tr -d ' \n')"
}

test_oversized_center

# scale=frame: the overlay should be re-rendered at the current frame
# dimensions. Use a 4×2 native PAM (a checkerboard, so a stretch is
# detectable from the colour at known coordinates) and a 48×4 frame.
# After scale=frame, the overlay covers the whole frame; sampling
# pixel (0,0) and (47,0) — both in row 0 — should yield the colour
# of the corresponding source column after the horizontal stretch.
test_scale_frame() {
    echo "==> scale=frame stretches small PAM to the full frame"
    local sdir=$WORK/scaleframe
    mkdir -p "$sdir"
    local pam=$sdir/tiny.pam
    # 4x2 PAM: row 0 = red, green, blue, white; row 1 = same.
    {
        printf 'P7\nWIDTH 4\nHEIGHT 2\nDEPTH 4\nMAXVAL 255\n'
        printf 'TUPLTYPE RGB_ALPHA\nENDHDR\n'
        for _ in 0 1; do
            printf '\xff\x00\x00\xff'   # red
            printf '\x00\xff\x00\xff'   # green
            printf '\x00\x00\xff\xff'   # blue
            printf '\xff\xff\xff\xff'   # white
        done
    } > "$pam"

    cd "$sdir"
    # dump:skip=4 lets the async rescale (kicked from postprocess on
    # the first frame) complete and harvest before we dump frame 5.
    # fps=10 keeps the wait short. dummy parses these as separate
    # colon-tokens — not `dump=skip=4` (that's parsed as just dump).
    "$UV" -t "testcard:size=${FRAME_W}x${FRAME_H}:fps=10:codec=RGBA:pattern=blank=0xFF" \
          -d "dummy:codec=RGBA:dump:skip=4:oneshot:raw" \
          --postprocess "overlay:file=$pam:scale=frame:scale_filter=nearest" \
          >/dev/null 2>&1 &
    local pid=$!
    local n=0
    while kill -0 "$pid" 2>/dev/null && [ $n -lt 25 ]; do sleep 0.2; n=$((n+1)); done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    cd - >/dev/null

    local f=$(ls "$sdir"/dummy.* 2>/dev/null | head -1)
    [ -s "$f" ] || { echo "  FAIL: no dump"; FAIL=$((FAIL+1)); return; }

    # 4-wide PAM stretched to 48 columns with nearest-neighbour: each
    # source column owns 12 frame columns. Pixel (0,0) is red,
    # pixel (47,0) is white.
    assert_eq "scale=frame: pixel(0,0) is red"   "ff0000ff" \
              "$(od -An -tx1 -N4 -j0 "$f" | tr -d ' \n')"
    assert_eq "scale=frame: pixel(47,0) is white" "ffffffff" \
              "$(od -An -tx1 -N4 -j$((47 * 4)) "$f" | tr -d ' \n')"
}

test_scale_frame

# I420 parallel-vs-serial parity. The frame and overlay are sized so the
# overlay rect (1280x720 = 921k pixels) crosses the MIN_PARALLEL_PIXELS
# = 500k threshold inside blend_i420 — without this the test would fall
# through to the serial path and prove nothing.
test_i420_parallel_parity() {
    echo "==> I420 parallel parity (1 vs 8 workers, 1280x720 overlay)"
    local pdir=$WORK/i420parity
    mkdir -p "$pdir"
    local big=$pdir/big.pam
    python3 - "$big" <<'PYEOF' || { echo "  python3 missing — skip"; return; }
import struct, sys
W, H = 1280, 720
with open(sys.argv[1], "wb") as f:
    f.write(f'P7\nWIDTH {W}\nHEIGHT {H}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'.encode())
    for y in range(H):
        for x in range(W):
            f.write(struct.pack('BBBB', x & 0xff, y & 0xff, (x ^ y) & 0xff, 0x80))
PYEOF
    run_threads() {
        local label=$1 threads=$2
        mkdir -p "$pdir/$label"
        cd "$pdir/$label"
        "$UV" -t "testcard:size=1920x1080:fps=1:codec=I420:pattern=ebu_bars" \
              -d "dummy:codec=I420:dump=oneshot:raw" \
              --postprocess "overlay:file=$big:position=top_left:blend_threads=$threads" \
              >/dev/null 2>&1 &
        local pid=$!
        local n=0
        while kill -0 "$pid" 2>/dev/null && [ $n -lt 30 ]; do sleep 0.2; n=$((n+1)); done
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        cd - >/dev/null
    }
    run_threads t1 1
    run_threads t8 8
    # cmp -s passes for two empty files; uv crashing in both runs would
    # silently pass without this guard.
    if [ ! -s "$pdir/t1/dummy.yuv" ] || [ ! -s "$pdir/t8/dummy.yuv" ]; then
        echo "  FAIL: I420 dump missing or empty (uv crash?)"
        FAIL=$((FAIL + 1))
        return
    fi
    if cmp -s "$pdir/t1/dummy.yuv" "$pdir/t8/dummy.yuv"; then
        PASS=$((PASS + 1))
    else
        echo "  FAIL: I420 serial vs parallel output differs"
        FAIL=$((FAIL + 1))
    fi
}

test_i420_parallel_parity

echo
echo "passed: $PASS   failed: $FAIL"
[ $FAIL -eq 0 ]
