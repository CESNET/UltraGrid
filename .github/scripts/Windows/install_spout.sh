#!/bin/sh -eux
# Install SPOUT

install() {(
        # .github/scripts/download-gh-asset.sh leadedge/Spout2 binaries Spout.zip
        curl -LSs https://github.com/leadedge/Spout2/releases/download/\
2.007.016/Spout-SDK-binaries_2-007-016.zip -o Spout.zip
        unzip Spout.zip
        # d=$(echo Spout-SDK-binaries/Libs_*)
        d=$(echo Spout-SDK-binaries/*/Libs)
        cp "$d"/MT/bin/SpoutLibrary.dll /usr/local/bin/
        cp "$d"/MT/lib/SpoutLibrary.lib /usr/local/lib/
        cp "$d"/include/SpoutLibrary/SpoutLibrary.h /usr/local/include/
)}

install
