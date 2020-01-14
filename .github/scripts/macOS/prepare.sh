#!/bin/bash

set -ex

echo "::set-env name=AJA_DIRECTORY::/var/tmp/ntv2sdk"
echo "::set-env name=UG_SKIP_NET_TESTS::1"
echo "::set-env name=CPATH::/usr/local/include:/usr/local/opt/qt/include"
echo "::set-env name=LIBRARY_PATH::/usr/local/lib:/usr/local/opt/qt/lib"
echo "::set-env name=PKG_CONFIG_PATH::/usr/local/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig"
echo "::set-env name=SDKROOT::macosx10.14" # SDK 10.15 crashes Qt in High Sierra
echo "::add-path::/usr/local/opt/qt/bin"

curl -LO https://github.com/phracker/MacOSX-SDKs/releases/download/10.14-beta4/MacOSX10.14.sdk.tar.xz
tar xJf MacOSX10.14.sdk.tar.xz
mv MacOSX10.14.sdk /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
brew install autoconf automake cppunit dylibbundler libtool pkg-config
brew install ffmpeg portaudio sdl2
brew install ossp-uuid # for cineform
( cd cineform-sdk/ && cmake . && make CFHDCodecStatic )

# Install XIMEA
if [ -z "$sdk_pass" ]; then exit 0; fi
curl --netrc-file <(cat <<<"machine frakira.fi.muni.cz login sdk password $sdk_pass") https://frakira.fi.muni.cz/~xpulec/sdks/m3api.tar.xz -O
sudo tar xJf m3api.tar.xz -C $(xcrun --show-sdk-path)/$SDKPATH/System/Library/Frameworks

# Install AJA
if [ -z "$sdk_pass" ]; then exit 0; fi
curl --netrc-file <(cat <<<"machine frakira.fi.muni.cz login sdk password $sdk_pass") https://frakira.fi.muni.cz/~xpulec/sdks/ntv2sdkmac.zip -O
unzip ntv2sdkmac.zip -d /var/tmp
mv /var/tmp/ntv2sdk* /var/tmp/ntv2sdk
cd /var/tmp/ntv2sdk/ajalibraries/ajantv2/build
xcodebuild -project ajantv2.xcodeproj
sudo rm -f /usr/local/lib/libajantv2.dylib
sudo cp ../../../bin/ajantv2.dylib /usr/local/lib/libajantv2.dylib
sudo ln -fs /usr/local/lib/libajantv2.dylib /usr/local/lib/ajantv2.dylib

