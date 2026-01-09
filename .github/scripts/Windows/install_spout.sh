#!/bin/sh -eux
# Install SPOUT

install() {(
        .github/scripts/download-gh-asset.sh leadedge/Spout2 binaries Spout.zip
        unzip Spout.zip
        d=$(echo Spout-SDK-binaries/Libs_*)
        cp "$d"/MT/bin/SpoutLibrary.dll /usr/local/bin/
        cp "$d"/MT/lib/SpoutLibrary.lib /usr/local/lib/
        cp "$d"/include/SpoutLibrary/SpoutLibrary.h /usr/local/include/
)}

install
