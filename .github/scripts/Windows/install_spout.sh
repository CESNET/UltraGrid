#!/bin/sh -eux
# Install SPOUT

git clone --depth 1 https://github.com/leadedge/Spout2.git
cp Spout2/BUILD/Binaries/x64/SpoutLibrary.dll /usr/local/bin
cp Spout2/BUILD/Binaries/x64/SpoutLibrary.lib /usr/local/lib
cp Spout2/SPOUTSDK/SpoutLibrary/SpoutLibrary.h /usr/local/include

