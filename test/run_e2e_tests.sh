#!/bin/sh -eux
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
#
# This scripts is called by .github/workflows/semi-weeekly_tests.yml.
# The idea is to do some further functional UltraGrid tests.

if [ "${1-}" = help ] || [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]
then
        echo "Usage:"
        printf "\t%s [<path_to_uv> <path_to_hd_rum_transcode> [release]]\n" "$0"
        exit 0
fi

origdir="$(realpath -- "$(dirname -- "$0")")"

# note: if run_uv/run_reflector given as a env var, those must be with
# the absolute path (see also further note)
if [ ! "${run_uv:=${1-}}" ]; then
        run_uv=./bin/uv
        run_uv=$(realpath "$run_uv")
fi
if [ ! "${run_reflector:=${2-}}" ]; then
        run_reflector=./bin/hd-rum-transcode
        run_reflector=$(realpath "$run_reflector")
fi
# note: intentionally not enclosing in "" - when invoked from run_scheduled_tests.sh
# in Linux, the values for reflector is ".../AppRun -o hd-rum-transcode"
if ! $run_uv -v >/dev/null 2>&1; then
        echo "uv binary $run_uv not found or cannot be executed"
        exit 2
fi
if ! $run_reflector -v >/dev/null 2>&1; then
        echo "hd-rum-transcode binary $run_reflector not found or cannot be executed"
        exit 2
fi
release=
if [ "${3-}" = release ]; then
        release=1
fi

# use empty dir to avoid need to clean between continuous and release
cd "$(mktemp -d)"

sys=$(uname -s)
if [ "$sys" = Linux ]; then
        OS=Linux
elif [ "$sys" = Darwin ]; then
        OS=macOS
else
        OS=Windows
fi

# seems like timeout is present natively since macOS 27
if [ "$OS" = macOS ] && ! command -v timeout >/dev/null; then
        timeout() {
                gtimeout "$@"
        }
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
. "$origdir"/run_e2e_tests_data.sh

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
                if ! expr -- "$opts" : ".*$OS" >/dev/null; then
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
 succeeded!\n" "$args" "$rc"
                        exit 1
                fi

        fi
done

printf "\n\n%s: all tests succeeded!\n\n" "$0"
