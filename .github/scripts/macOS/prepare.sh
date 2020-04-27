#!/bin/bash -eu

AJA_INST=/var/tmp/ntv2sdk # AJA installation directory
TEMP_INST=/tmp/install
export SDKROOT=macosx10.14

echo "::set-env name=AJA_DIRECTORY::$AJA_INST"
echo "::set-env name=UG_SKIP_NET_TESTS::1"
echo "::set-env name=CPATH::/usr/local/include:/usr/local/opt/qt/include"
echo "::set-env name=EXTRA_LIB_PATH::/usr/local/lib" # workaround for dylibbunder inside Makefile (DYLD_LIBRARY_PATH is stripped
                                                     # by make) to resolve paths like @loader_path/ (see https://github.com/auriamg/macdylibbundler/issues/22)
echo "::set-env name=LIBRARY_PATH::/usr/local/lib:/usr/local/opt/qt/lib"
# libcrypto.pc (and other libcrypto files) is not linked to /usr/local/{lib,include} because conflicting with system libcrypto
echo "::set-env name=PKG_CONFIG_PATH::/usr/local/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig:/usr/local/opt/openssl/lib/pkgconfig"
echo "::set-env name=SDKROOT::$SDKROOT" # SDK 10.15 crashes Qt in High Sierra
echo "::add-path::/usr/local/opt/qt/bin"

curl -LO https://github.com/phracker/MacOSX-SDKs/releases/download/10.14-beta4/${SDKROOT}.sdk.tar.xz
tar xJf ${SDKROOT}.sdk.tar.xz
mv ${SDKROOT}.sdk /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs
rm ${SDKROOT}.sdk.tar.xz
brew install autoconf automake cppunit dylibbundler libtool pkg-config
brew install ffmpeg portaudio sdl2
brew install imagemagick jack opencv openssl
brew install ossp-uuid # for cineform
( cd cineform-sdk/ && cmake . && make CFHDCodecStatic )
brew install qt

mkdir $TEMP_INST
cd $TEMP_INST

# Install XIMEA
if [ -n "$sdk_pass" ]; then
        curl --netrc-file <(cat <<<"machine frakira.fi.muni.cz login sdk password $sdk_pass") https://frakira.fi.muni.cz/~xpulec/sdks/m3api.tar.xz -O
        sudo tar xJf m3api.tar.xz -C $(xcrun --show-sdk-path)/System/Library/Frameworks
fi

# Install AJA
if [ -n "$sdk_pass" ]; then
        curl --netrc-file <(cat <<<"machine frakira.fi.muni.cz login sdk password $sdk_pass") https://frakira.fi.muni.cz/~xpulec/sdks/ntv2sdkmac.zip -O
        unzip ntv2sdkmac.zip -d /tmp
        mv /tmp/ntv2sdk* $AJA_INST
        cd $AJA_INST/ajalibraries/ajantv2/build
        xcodebuild -project ajantv2.xcodeproj
        sudo rm -f /usr/local/lib/libajantv2.dylib
        sudo cp ../../../bin/ajantv2.dylib /usr/local/lib/libajantv2.dylib
        sudo ln -fs /usr/local/lib/libajantv2.dylib /usr/local/lib/ajantv2.dylib
        cd $TEMP_INST
fi

# Install live555
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles macosx
make install
cd ..

# Install Syphon
wget --no-verbose https://github.com/Syphon/Syphon-Framework/releases/download/5/Syphon.SDK.5.zip
unzip Syphon.SDK.5.zip
sudo cp -R 'Syphon SDK 5/Syphon.framework' /Library/Frameworks

# Remove installation files
cd
rm -rf $TEMP_INST

