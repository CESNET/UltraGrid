#!/bin/bash

set -v

#export PATH='/usr/local/bin:/usr/bin:/bin'

function run_in_vs_env
{
    eval vssetup=\$"$1"'\\..\\..\\VC\\bin\\amd64\\vcvars64.bat'
    cmd //Q //C call "${vssetup?vssetup not set!}" "&&" "${@:2}"
}

function run_vs16
{
    eval vssetup='C:\\Program\ Files\ \(x86\)\\Microsoft\ Visual\ Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat'
    cmd //Q //C call "${vssetup?vssetup not set!}" "&&" "$@"
}

function run_vs12
{
    run_in_vs_env VS120COMNTOOLS "$@"
}


function run_vs11
{
    run_in_vs_env VS110COMNTOOLS "$@"
}

function run_vs10
{
    run_in_vs_env VS100COMNTOOLS "$@"
}

# shellcheck disable=SC2016
export INCLUDE='src;AJA\ajalibraries\ajantv2\includes;AJA\ajalibraries\ajantv2\src\win;AJA\ajaapps\crossplatform\demoapps;AJA\ajalibraries;AJA\ajaapps\crossplatform\demoapps\ntv2capture;$INCLUDE'

# rename both aja.cpp because msvc creates object aja.obj for both in the same directory
cp src/video_capture/aja.cpp aja_capture.cpp
cp src/video_display/aja.cpp aja_display.cpp

MSVS_PATH=$(/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer/vswhere.exe -latest -property installationPath)

eval vssetup=\""$MSVS_PATH"\"'\\VC\\Auxiliary\\Build\\vcvars64.bat'
cmd //Q //C call "$vssetup" "&&" cl //std:c++latest //EHsc //LD //D_XKEYCHECK_H //DAJA_WINDOWS //DMSWindows //DAJA_NTV2SDK_VERSION_MAJOR=13 src/aja_common.cpp aja_capture.cpp aja_display.cpp src/video_capture/aja_win32_utils.cpp src/video_capture_params.cpp src/utils/config_file.cpp src/utils/video_frame_pool.cpp AJA/lib/ajantv2_16.lib advapi32.lib Netapi32.lib Shell32.lib Shlwapi.lib user32.lib winmm.lib //Feaja

cp aja.lib /usr/local/lib
cp aja.dll /usr/local/bin

