#!/bin/bash

#######################################
# USAGE
#
# 1) Copy SpoutSDK to src
# 2) run build_spout.sh (this script)
#######################################

function run_in_vs_env
{
    eval vssetup="\$$1\\vsvars32.bat"
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

run_vs12 cl //DEXPORT_DLL_SYMBOLS src/spout_sender.cpp //LD src/SpoutSDK/VS2012/Binaries/Win32/Spout.lib //Fospout_sender
cp spout_sender.dll /usr/local/bin
cp spout_sender.lib /usr/local/lib

