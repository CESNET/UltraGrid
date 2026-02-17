#!/bin/sh -eux
# Install SPOUT

build() {(
        cd /c
        rm -rf Spout2
        git clone --depth 1 https://github.com/leadedge/Spout2.git
        cd Spout2

        # do not cherry-pick - we have shallow clones
        git fetch --depth 2 https://github.com/MartinPulec/Spout2
        git format-patch -1 FETCH_HEAD --stdout > patch.diff
        git am < patch.diff

        /c/Program\ Files/CMake/bin/cmake.exe -Bbuild2 . # ./BUILD already exists
        /c/Program\ Files/CMake/bin/cmake.exe --build build2 --config Release \
         -j "$(nproc)"
)}

install() {(
        mkdir -p /usr/local/bin /usr/local/include /usr/local/lib
        cp /c/Spout2/build2/bin/Release/SpoutLibrary.dll /usr/local/bin/
        cp /c/Spout2/build2/lib/Release/SpoutLibrary.lib /usr/local/lib/
        cp /c/Spout2/SPOUTSDK/SpoutLibrary/SpoutLibrary.h /usr/local/include/
)}

$1
