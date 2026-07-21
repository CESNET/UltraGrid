#!/bin/sh -eu
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
#
# This scripts is called by .github/workflows/semi-weeekly_tests.yml.
# The idea is to do some further functional UltraGrid tests.

release=
if [ "${1-}" = release ]; then
        release=1
fi

origdir=$(pwd)
# use empty dir to avoid need to clean between continuous and release
cd "$(mktemp -d)"

case "${RUNNER_OS-}" in
        Linux)
                sudo apt update
                ## @sa (taken from) .github/scripts/Linux/utils/Dockerfile.ubuntu
                sudo apt -y install libasound2t64 libegl1 libfontconfig1 \
                    libglx-mesa0 libgmp10 libharfbuzz0b libopengl0 libp11-kit0 \
                    libx11-6 openbox xvfb
                Xvfb :99 -screen 0 1920x1080x24 &
                export DISPLAY=:99
                openbox &

                continuous_build_file=UltraGrid-continuous-x86_64.AppImage
                release_pattern=AppImage
                prepare() {
                        ai=$(echo UltraGrid-*.AppImage)
                        chmod +x "$ai"
                        ./"$ai" --appimage-extract
                }
                run_uv=squashfs-root/AppRun
                run_reflector="squashfs-root/AppRun -o hd-rum-transcode"
                ;;
        Windows)
                continuous_build_file=UltraGrid-continuous-win64.zip
                release_pattern=win64
                prepare() {
                        unzip UltraGrid-*-win64.zip
                }
                run_uv=$(echo UltraGrid-*-win64/uv.exe)
                run_reflector=$(echo UltraGrid-*-win64/hd-rum-transcode.exe)
                ;;
        macOS)
                continuous_build_file=UltraGrid-continuous-arm64.dmg
                release_pattern=arm64
                prepare() {
                        hdiutil mount UltraGrid-*arm64.dmg
                }
                brew install coreutils
                alias timeout=gtimeout
                run_uv=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/uv
                run_reflector=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/\
hd-rum-transcode
esac

if [ "${RUNNER_OS-}" ]; then
        if [ "$release" ]; then
                "$origdir"/.github/scripts/download-gh-asset.sh \
                        CESNET/UltraGrid '.*'"${release_pattern?}"'.*'
        else
                curl -LOf https://github.com/CESNET/UltraGrid/releases/\
download/continuous/"${continuous_build_file?}"
        fi
        prepare
else # for INTERACTIVE use only (debugging)
        if [ "${1-}" = help ] || [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]
        then
                echo "Usage (interactive only, not CI):"
                printf "\t%s <path_to_uv> <path_to_hd_rum_transcode>\n" "$0"
                exit 0
        fi
        run_uv=${1?Please provide path to uv as first argument}
        run_reflector=${2?Please provide path to hd-rum-trancode as second argument}
        sys=$(uname -s)
        if [ "$sys" = Linux ]; then
                RUNNER_OS=Linux
        elif [ "$sys" = Darwin ]; then
                RUNNER_OS=macOS
        else
                RUNNER_OS=Windows
        fi
fi

## used by run_test_data.sh
## @param $1 args
## @param $2 opts (optional, separated by comma):
##  - should_fail - expected outcome of the command is failure
##  - should_timeout - the command is expected to keep running
##                     (will be terminated by timeout)
##  - run_reflector - instead of uv, pass the args to hd-rum-transcode
##  - Linux_only    - run just in Linux   runner
##  - Windows_only  - "   "    "  Windows "
##  - macOS_only    - "   "    "  macOS   "
##
## More platforms with "_only" suffix can be specified, however, eg.
## Linux_only,macOS_only.
add_test() {
        eval "test_${test_count}_args=\${1?}"
        eval "test_${test_count}_opts=\${2-}"
        test_count=$((test_count + 1))
}
## checks options for unknown keyword (to avoid typo)
validate_opts() {
        n_opts=$(echo "${1?}" | sed 's/,/ /g')
        # shellcheck disable=SC2086 # intentional
        set -- $n_opts
        while [ $# -gt 0 ]; do
                if [ "$1" != run_reflector ] &&
                   [ "$1" != should_fail ] &&
                   [ "$1" != should_timeout ] &&
                   [ "$1" != continuous_only ] &&
                   [ "$1" != Linux_only ] &&
                   [ "$1" != Windows_only ] &&
                   [ "$1" != macOS_only ]; then
                        echo "Wrong option $1!" 1>&2
                        exit 10
                fi
                shift
        done
}

test_count=0
. "$origdir"/.github/scripts/run_scheduled_tests_data.sh

set +e
i=0
while [ $i -lt $test_count ]; do
        eval "args=\$test_${i}_args"
        eval "opts=\$test_${i}_opts"

        # shellcheck disable=SC2154 # set by eval
        validate_opts "$opts"

        i=$((i + 1))

        exec=$run_uv
        tool=uv
        if expr -- "$opts" : '.*run_reflector' >/dev/null; then
                tool=reflector
                exec=$run_reflector
        fi
        # skip one-platform only tests if we are not the target
        if expr -- "$opts" : '.*_only' >/dev/null; then
                if ! expr -- "$opts" : ".*$RUNNER_OS" >/dev/null; then
                        continue
                fi
        fi
        if [ "$release" ] && expr -- "$opts" : '.*continuous_only' >/dev/null; then
                continue
        fi

        timeout=10
        # shellcheck disable=SC2154 # set by eval
        echo "Starting: \"timeout $timeout $exec $args\""
        # shellcheck disable=SC2086 # intentional - split words
        timeout -k $((timeout+5)) $timeout $exec $args
        rc=$?
        echo "Finished: \"timeout $timeout $exec $args\" with RC $rc"

        if [ $rc = 124 ]; then
                if ! expr -- "$opts" : '.*should_timeout' >/dev/null; then
                        printf "$tool with arguments %s timeout (limit: %d sec)!\n" \
                                "$args" "$timeout"
                        exit 1
                fi
        elif expr -- "$opts" : '.*should_fail' >/dev/null; then
                if [ $rc -eq 0 ]; then
                        printf "$tool with arguments %s should have failed but\
 returned 0!\n" "$args"
                        exit 1
                fi
        else
                if [ $rc -ne 0 ]; then
                        printf "$tool with arguments %s returned %d but should have\
 succeeeded!\n" "$args" "$rc"
                        exit 1
                fi

        fi
done

printf "\n\n%s: all tests succeeded!\n\n" "$0"
