#!/bin/sh -eu
#
# THis scripts is called by .github/workflows/semi-weeekly_tests.yml
# The idea is to do some further functional UltraGrid tests

# see also (taken from) .github/scripts/Linux/utils/Dockerfile.ubuntu
ubuntu_packages="
    libasound2t64\
    libegl1\
    libfontconfig1\
    libglx-mesa0\
    libgmp10\
    libharfbuzz0b\
    libopengl0\
    libp11-kit0\
    libx11-6\
    openbox\
    xvfb"

case "$RUNNER_OS" in
        Linux)
                sudo apt update
                sudo apt -y install $ubuntu_packages
                Xvfb :99 -screen 0 1920x1080x24 &
                export DISPLAY=:99
                openbox &

                ug_build=UltraGrid-continuous-x86_64.AppImage
                prepare() {
                        chmod +x "$ug_build"
                        ./"$ug_build" --appimage-extract
                }
                run_uv=squashfs-root/AppRun
                run_reflector="squashfs-root/AppRun -o hd-rum-transcode"
                ;;
        Windows)
                ug_build=UltraGrid-continuous-win64.zip
                prepare() {
                        unzip "$ug_build"
                }
                run_uv=UltraGrid-continuous-win64/uv.exe
                run_reflector=UltraGrid-continuous-win64/hd-rum-transcode.exe
                ;;
        macOS)
                ug_build=UltraGrid-continuous-arm64.dmg
                prepare() {
                        hdiutil mount "$ug_build"
                }
                brew install coreutils
                alias timeout=gtimeout
                run_uv=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/uv
                run_reflector=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/\
hd-rum-transcode
esac

curl -LOf https://github.com/CESNET/UltraGrid/releases/download/continuous/\
"$ug_build"

prepare

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

test_count=0
. "$(dirname "$0")"/run_scheduled_tests_data.sh

set +e
i=0
while [ $i -lt $test_count ]; do
        eval "args=\$test_${i}_args"
        eval "opts=\$test_${i}_opts"

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

        timeout=5
        echo "Starting: \"timeout $timeout $exec $args\""
        timeout -k 10 $timeout $exec $args
        rc=$?
        echo "Finished: \"timeout $timeout $exec $args\" with RC %rc"

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

