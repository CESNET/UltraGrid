#!/bin/bash -eu

AJA_INST=/var/tmp/ntv2sdk # AJA installation directory
TEMP_INST=/tmp/install

CPATH=/usr/local/include:/usr/local/opt/qt/include
LIBRARY_PATH=/usr/local/lib:/usr/local/opt/qt/lib
echo "AJA_DIRECTORY=$AJA_INST" >> $GITHUB_ENV
echo "UG_SKIP_NET_TESTS=1" >> $GITHUB_ENV
echo "CPATH=$CPATH" >> $GITHUB_ENV
echo "LIBRARY_PATH=$LIBRARY_PATH" >> $GITHUB_ENV
# libcrypto.pc (and other libcrypto files) is not linked to /usr/local/{lib,include} because conflicting with system libcrypto
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig:/usr/local/opt/openssl/lib/pkgconfig" >> $GITHUB_ENV
echo "/usr/local/opt/qt/bin" >> $GITHUB_PATH

brew install autoconf automake cppunit libtool pkg-config
brew install ffmpeg portaudio sdl2
brew install imagemagick jack opencv openssl
brew install ossp-uuid # for cineform
( cd cineform-sdk/ && cmake -DBUILD_TOOLS=OFF . && make CFHDCodecStatic || exit 1 )
brew install qt

.github/scripts/macOS/install_dylibbundler_v2.sh

mkdir $TEMP_INST
cd $TEMP_INST

# Install XIMEA
if [ -f /var/tmp/sdks/m3api.tar.xz ]; then
        sudo tar xJf /var/tmp/sdks/m3api.tar.xz -C $(xcrun --show-sdk-path)/System/Library/Frameworks
fi

# Install AJA
if [ -f /var/tmp/sdks/ntv2sdkmac.zip ]; then
        unzip /var/tmp/sdks/ntv2sdkmac.zip -d /tmp
        mv /tmp/ntv2sdk* $AJA_INST
        cd $AJA_INST/ajalibraries/ajantv2/build
        xcodebuild -project ajantv2.xcodeproj
        sudo rm -f /usr/local/lib/libajantv2.dylib
        sudo cp ../../../bin/ajantv2.dylib /usr/local/lib/libajantv2.dylib
        sudo ln -fs /usr/local/lib/libajantv2.dylib /usr/local/lib/ajantv2.dylib
        cd $TEMP_INST
fi

# DELTACAST
if [ -f /var/tmp/sdks/VideoMasterHD_mac.tar.xz ]; then
        sudo tar xJf /var/tmp/sdks/VideoMasterHD_mac.tar.xz -C $(xcrun --show-sdk-path)/System/Library/Frameworks
fi

# Install NDI
if [ -f /var/tmp/sdks/NDISDK_Apple.pkg  ]; then
        sudo installer -pkg /var/tmp/sdks/NDISDK_Apple.pkg -target /
        sudo mv "/Library/NDI SDK for Apple/" /Library/NDI
        cd /Library/NDI/lib/x64
        sudo ln -s libndi.?.dylib libndi.dylib
        export CPATH=${CPATH:+"$CPATH:"}/Library/NDI/include
        export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s /Library/NDI/lib/x64"
        export LIBRARY_PATH=${LIBRARY_PATH:+"$LIBRARY_PATH:"}/Library/NDI/lib/x64
        export MY_DYLD_LIBRARY_PATH="${MY_DYLD_LIBRARY_PATH:+$MY_DYLD_LIBRARY_PATH:}/Library/NDI/lib/x64"
        echo "CPATH=$CPATH" >> $GITHUB_ENV
        echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> $GITHUB_ENV
        echo "LIBRARY_PATH=$LIBRARY_PATH" >> $GITHUB_ENV
        echo "MY_DYLD_LIBRARY_PATH=$MY_DYLD_LIBRARY_PATH" >> $GITHUB_ENV
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

# Install cross-platform deps
$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh

# Remove installation files
cd
rm -rf $TEMP_INST

