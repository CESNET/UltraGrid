#!/bin/bash -eu

#######################################
# USAGE
#
# 1) Copy SpoutSDK to src
# 2) run build_spout.sh (this script)
#######################################

function run_in_vs_env
{
    eval vssetup=\$$1'\\..\\..\\VC\\bin\\amd64\\vcvars64.bat'
    cmd //Q //C call "$vssetup" "&&" "${@:2}"
}

function run_vs16
{
    eval vssetup='C:\\Program\ Files\ \(x86\)\\Microsoft\ Visual\ Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat'
    cmd //Q //C call "$vssetup" "&&" "$@"
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

LIBDIR=${1:-src/SpoutSDK/Binaries/x64}
MSVS_PATH=`/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer/vswhere.exe -latest -property installationPath`

eval vssetup=\"$MSVS_PATH\"'\\VC\\Auxiliary\\Build\\vcvars64.bat'
cmd //Q //C call "$vssetup" "&&" cl //DEXPORT_DLL_SYMBOLS src/spout_sender.cpp src/spout_receiver.cpp //LD $LIBDIR/Spout.lib //Fespout_wrapper

cp spout_wrapper.dll /usr/local/bin
cp spout_wrapper.lib /usr/local/lib
cp $LIBDIR/Spout.dll /usr/local/bin

