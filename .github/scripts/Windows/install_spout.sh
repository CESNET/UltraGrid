#!/bin/sh -eux
# Install SPOUT

git clone -b 2.006 --depth 1 https://github.com/leadedge/Spout2.git
mkdir Spout2/SpoutSDK/Source/build
cd Spout2/SpoutSDK/Source/build
cmake -DBUILD_SHARED_LIBS=ON -G 'MSYS Makefiles' ..
cmake --build . --parallel
cp libSpout.dll /usr/local/bin
cp libSpout.dll.a /usr/local/lib
cd -
mkdir /usr/local/include/SpoutSDK
cp Spout2/SpoutSDK/Source/*.h /usr/local/include/SpoutSDK
rm -rf Spout2

