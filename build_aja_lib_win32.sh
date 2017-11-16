#!/bin/bash

set -v

export INCLUDE='src;C:\msys32\home\host\AJA\APIandSamples\ntv2projects\classes;C:\msys32\home\host\AJA\APIandSamples\ntv2projects\includes;C:\msys32\home\host\AJA\APIandSamples\ntv2projects\winclasses;C:\msys32\home\host\AJA\APIandSamples\ntv2projects\democlasses;C:\msys32\home\host\AJA\APIandSamples\ajaapi;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\INCLUDE;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\ATLMFC\INCLUDE;C:\Program Files (x86)\Windows Kits\8.1\include\shared;C:\Program Files (x86)\Windows Kits\8.1\include\um;C:\Program Files (x86)\Windows Kits\8.1\include\winrt;'
export LIBPATH='C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\LIB;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\ATLMFC\LIB;C:\Program Files (x86)\Windows Kits\8.1\References\CommonConfiguration\Neutral;C:\Program Files (x86)\Microsoft SDKs\Windows\v8.1\ExtensionSDKs\Microsoft.VCLibs\12.0\References\CommonConfiguration\neutral;'
export PATH='/usr/local/bin:/usr/bin:/bin:/c/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/'
export LIB='C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\LIB;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\ATLMFC\LIB;C:\Program Files (x86)\Windows Kits\8.1\lib\winv6.3\um\x86;'
cl //LD //DAJA_WINDOWS //DMSWindows //DAJA_NTV2SDK_VERSION_MAJOR=13 src/video_capture/aja.cpp src/video_capture/aja_win32_utils.cpp src/video_capture_params.cpp src/utils/config_file.cpp ../AJA/APIandSamples/lib/ajastuff.lib ../AJA/APIandSamples/lib/classesSTATIC.lib advapi32.lib user32.lib
cp aja.lib /usr/local/lib
cp aja.dll /usr/local/bin
cp aja.dll bin

