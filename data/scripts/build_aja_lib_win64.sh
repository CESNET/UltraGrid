#!/bin/bash -ex

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

if [ -d libajantv2 ]; then
    AJA_PREF=.
else
    AJA_PREF=$(cygpath -w "$HOME")
fi
export INCLUDE="src;$AJA_PREF\libajantv2;$AJA_PREF\libajantv2\ajantv2\includes;\
$AJA_PREF\libajantv2\ajantv2\src\win"

# rename both aja.cpp because msvc creates object aja.obj for both in the same directory
cp src/video_capture/aja.cpp aja_capture.cpp
cp src/video_display/aja.cpp aja_display.cpp

MSVS_PATH=$(/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer/vswhere.exe -latest -property installationPath)

eval vssetup=\""$MSVS_PATH"\"'\\VC\\Auxiliary\\Build\\vcvars64.bat'
cmd //Q //C call "$vssetup" "&&" cl //std:c++latest //EHsc //LD //D_XKEYCHECK_H \
      //DAJA_WINDOWS //DMSWindows //DAJA_NTV2SDK_VERSION_MAJOR=13 \
      aja_capture.cpp aja_display.cpp src/aja_common.cpp \
      //c //MD //d2FH4-
# d2FH4- -> https://stackoverflow.com/a/70448867, otherwise linking
# with -lvcruntime would be required; it is distributed inside MSVS so
# also paths would need to be set then

cp aja_common.obj aja_capture.obj aja_display.obj /usr/local/lib
cp "$AJA_PREF"/libajantv2/build/ajantv2/Release/ajantv2*.lib /usr/local/lib
cp "$AJA_PREF"/libajantv2/build/ajantv2/Release/ajantv2*.dll /usr/local/bin

