#!/bin/sh -eux
# Install SPOUT

git clone --depth 1 https://github.com/leadedge/Spout2.git
cd Spout2
/c/Program\ Files/CMake/bin/cmake.exe -Bbuild2 . # ./BUILD already exists
/c/Program\ Files/CMake/bin/cmake.exe --build build2 --parallel
mkdir -p /usr/local/bin /usr/local/include /usr/local/lib
cp build2/Binaries/x64/SpoutLibrary.dll /usr/local/bin
cp build2/Binaries/x64/SpoutLibrary.lib /usr/local/lib
cp SPOUTSDK/SpoutLibrary/SpoutLibrary.h /usr/local/include

