#!/bin/sh -eux
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
                run_uv=$(pwd)/squashfs-root/AppRun
                run_reflector="$(pwd)/squashfs-root/AppRun -o hd-rum-transcode"
                ;;
        Windows)
                continuous_build_file=UltraGrid-continuous-win64.zip
                release_pattern=win64
                prepare() {
                        unzip UltraGrid-*-win64.zip
                        run_uv=$(echo UltraGrid-*-win64/uv.exe)
                        run_reflector=$(echo UltraGrid-*-win64/hd-rum-transcode.exe)
                        run_uv=$(realpath "$run_uv")
                        run_reflector=$(realpath "$run_reflector")
                }
                ;;
        macOS)
                export HOMEBREW_GITHUB_API_TOKEN="$GITHUB_TOKEN"
                continuous_build_file=UltraGrid-continuous-arm64.dmg
                release_pattern=arm64
                prepare() {
                        hdiutil mount UltraGrid-*arm64.dmg
                }
                brew install coreutils
                run_uv=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/uv
                run_reflector=/Volumes/ULTRAGRID/uv-qt.app/Contents/MacOS/\
hd-rum-transcode
esac

if [ "$release" ]; then
        "$origdir"/.github/scripts/download-gh-asset.sh \
                CESNET/UltraGrid '.*'"${release_pattern?}"'.*'
else
        curl -LOf -H "Authorization: token $GITHUB_TOKEN" \
https://github.com/CESNET/UltraGrid/releases/\
download/continuous/"${continuous_build_file?}"
fi
prepare

"$origdir"/test/run_e2e_tests.sh "$run_uv" "$run_reflector" ${release:+release}
