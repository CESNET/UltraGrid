#!/bin/bash

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

run_vs12 cl //DEXPORT_DLL_SYMBOLS src/spout_sender.cpp src/spout_receiver.cpp //LD src/SpoutSDK/Binaries/x64/Spout.lib //Fespout_wrapper
cp spout_wrapper.dll /usr/local/bin
cp spout_wrapper.lib /usr/local/lib

