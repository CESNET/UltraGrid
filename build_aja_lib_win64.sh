#!/bin/bash

set -v

#export PATH='/usr/local/bin:/usr/bin:/bin'

function run_in_vs_env
{
    eval vssetup=\$$1'\\..\\..\\VC\\bin\\amd64\\vcvars64.bat'
    cmd //Q //C call "$vssetup" "&&" "${@:2}"
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

export INCLUDE='src;C:\msys64\home\toor\AJA\ajalibraries\ajantv2\includes;C:\msys64\home\toor\AJA\ajalibraries\ajantv2\src\win;C:\msys64\home\toor\AJA\ajaapps\crossplatform\demoapps;C:\msys64\home\toor\AJA\ajalibraries;C:\msys64\home\toor\AJA\ajaapps\crossplatform\demoapps\ntv2capture;$INCLUDE'

# rename both aja.cpp because msvc creates object aja.obj for both in the same directory
cp src/video_capture/aja.cpp aja_capture.cpp
cp src/video_display/aja.cpp aja_display.cpp

run_vs12 cl //LD //DAJA_WINDOWS //DMSWindows //DAJA_NTV2SDK_VERSION_MAJOR=13 aja_capture.cpp aja_display.cpp src/video_capture/aja_win32_utils.cpp src/video_capture_params.cpp src/utils/config_file.cpp ../AJA/lib/libajantv2.lib ../AJA/lib/libajabase.lib advapi32.lib user32.lib //Feaja
cp aja.lib /usr/local/lib
cp aja.dll /usr/local/bin

