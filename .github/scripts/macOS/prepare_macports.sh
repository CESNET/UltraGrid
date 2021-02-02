#!/bin/bash -eux

MACPORT_INST_CURR=MacPorts-2.6.4_1-11-BigSur.pkg
curl -LO https://distfiles.macports.org/MacPorts/$MACPORT_INST_CURR
sudo installer -pkg $MACPORT_INST_CURR -target /
echo '+universal' | sudo tee > /opt/local/etc/macports/variants.conf

AJA_INST=/var/tmp/ntv2sdk # AJA installation directory
TEMP_INST=/tmp/install

# following doesn't download universal binaries:
# $ file ~/Qt/6.0.3/clang_64/lib/QtGui.framework/Versions/A/QtGui
# /Users/xpulec/Qt/6.0.3/clang_64/lib/QtGui.framework/Versions/A/QtGui: Mach-O 64-bit dynamically linked shared library x86_64
# curl -LO 'https://d13lb3tujbc8s0.cloudfront.net/onlineinstallers/qt-unified-mac-x64-4.0.1-1-online.dmg'
# hdiutil mount qt-unified-mac-x64-4.0.1-1-online.dmg
# /Volumes/qt-unified-macOS-x86_64-4.0.1-1-online/qt-unified-macOS-x86_64-4.0.1-1-online.app/Contents/MacOS/qt-unified-macOS-x86_64-4.0.1-1-online --accept-licenses --da --ao --confirm-command -m XXXXX@YYYYY.ZZ --pw XXXXXXXXXXXXXX install preview.qt.qt6.61.clang_64

PATH=/opt/local/bin:$PATH
CPATH=/opt/local/include:/usr/local/include
LIBRARY_PATH=/opt/local/lib:/usr/local/lib
LD_LIBRARY_PATH=$LIBRARY_PATH
echo "AJA_DIRECTORY=$AJA_INST" >> $GITHUB_ENV
echo "UG_SKIP_NET_TESTS=1" >> $GITHUB_ENV
echo "CPATH=$CPATH" >> $GITHUB_ENV
echo "LIBRARY_PATH=$LIBRARY_PATH" >> $GITHUB_ENV
# libcrypto.pc (and other libcrypto files) is not linked to /usr/local/{lib,include} because conflicting with system libcrypto
#echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig:/usr/local/opt/openssl/lib/pkgconfig" >> $GITHUB_ENV
echo "/opt/local/bin" >> $GITHUB_PATH

INSTALL="sudo port -N install"
eval $INSTALL autoconf automake cppunit libtool pkgconfig
eval $INSTALL portaudio libsdl2
eval $INSTALL openssl
# ffmpeg imagemagick jack opencv
# eval $INSTALL ossp-uuid # for cineform
( cd cineform-sdk/ && cmake -DBUILD_TOOLS=OFF . && make CFHDCodecStatic || exit 1 )
# eval $INSTALL qt5

.github/scripts/macOS/install_dylibbundler_v2.sh

mkdir $TEMP_INST
cd $TEMP_INST

# Install XIMEA (see <dmg>/install.app/Contents/MacOS/install.sh)
hdiutil mount /var/tmp/sdks-free/XIMEA_OSX_SP.dmg
sudo cp -a /Volumes/XIMEA/m3api.framework $(xcrun --show-sdk-path)/System/Library/Frameworks
sudo xattr -dr com.apple.quarantine $(xcrun --show-sdk-path)/System/Library/Frameworks
umount /Volumes/XIMEA

# Install AJA
if [ -f /var/tmp/sdks/ntv2sdkmac.zip ]; then
        unzip /var/tmp/sdks/ntv2sdkmac.zip -d /tmp
        mv /tmp/ntv2sdk* $AJA_INST
        cd $AJA_INST/ajalibraries/ajantv2/build
        xcodebuild -project ajantv2.xcodeproj -arch x86_64 -arch arm64 ONLY_ACTIVE_ARCH=NO
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

