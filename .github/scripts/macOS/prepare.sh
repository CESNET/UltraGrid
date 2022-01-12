#!/bin/bash -eux

TEMP_INST=/tmp/install

CPATH=/usr/local/include:/usr/local/opt/qt/include
LIBRARY_PATH=/usr/local/lib:/usr/local/opt/qt/lib
echo "UG_SKIP_NET_TESTS=1" >> $GITHUB_ENV
echo "CPATH=$CPATH" >> $GITHUB_ENV
echo "LIBRARY_PATH=$LIBRARY_PATH" >> $GITHUB_ENV
# libcrypto.pc (and other libcrypto files) is not linked to /usr/local/{lib,include} because conflicting with system libcrypto
echo "PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/opt/qt/lib/pkgconfig:/usr/local/opt/openssl/lib/pkgconfig" >> $GITHUB_ENV
echo "/usr/local/opt/qt/bin" >> $GITHUB_PATH

brew install autoconf automake cppunit libtool pkg-config
brew install speexdsp
brew install ffmpeg portaudio sdl2
brew install imagemagick jack libnatpmp opencv openssl
brew install ossp-uuid # for cineform
( git submodule update --init cineform-sdk && cd cineform-sdk/ && cmake -DBUILD_TOOLS=OFF . && make -j $(sysctl -n hw.ncpu) CFHDCodecStatic || exit 1 )
brew install qt@5

sudo ln -s /usr/local/opt/qt@5 /usr/local/opt/qt

.github/scripts/macOS/install_dylibbundler_v2.sh

mkdir $TEMP_INST
cd $TEMP_INST

# Install XIMEA (see <dmg>/install.app/Contents/MacOS/install.sh)
install_ximea() {
        hdiutil mount /private/var/tmp/XIMEA_OSX_SP.dmg
        sudo cp -a /Volumes/XIMEA/m3api.framework $(xcrun --show-sdk-path)/System/Library/Frameworks
        sudo xattr -dr com.apple.quarantine $(xcrun --show-sdk-path)/System/Library/Frameworks
        umount /Volumes/XIMEA
}

# Install AJA
AJA_DIRECTORY=$SDK_NONFREE_PATH/ntv2sdk
if [ -d $AJA_DIRECTORY ]; then
        echo "AJA_DIRECTORY=$AJA_DIRECTORY" >> $GITHUB_ENV
        FEATURES="$FEATURES --enable-aja"
        echo "FEATURES=$FEATURES" >> $GITHUB_ENV
        sudo rm -f /usr/local/lib/libajantv2.dylib
        sudo cp $AJA_DIRECTORY/bin/ajantv2.dylib /usr/local/lib/libajantv2.dylib
        sudo ln -fs /usr/local/lib/libajantv2.dylib /usr/local/lib/ajantv2.dylib
        cd $TEMP_INST
fi

# DELTACAST
DELTA_CACHE_INST=$SDK_NONFREE_PATH/VideoMasterHD_inst
if [ -d $DELTA_CACHE_INST ]; then
        FEATURES="$FEATURES --enable-deltacast"
        echo "FEATURES=$FEATURES" >> $GITHUB_ENV
        sudo cp -a $DELTA_CACHE_INST/* $(xcrun --show-sdk-path)/System/Library/Frameworks
fi

# Install NDI
install_ndi() {
        sudo installer -pkg /private/var/tmp/Install_NDI_SDK_Apple.pkg -target /
        sudo mv /Library/NDI\ SDK\ for\ * /Library/NDI
        cat /Library/NDI/Version.txt | sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' | sudo tee /usr/local/include/ndi_version.h
        if [ -d /Library/NDI/lib/x64 ]; then # NDI 4
                cd /Library/NDI/lib/x64
                sudo ln -s libndi.?.dylib libndi.dylib
                NDI_LIB=/Library/NDI/lib/x64
        else # NDI 5
                NDI_LIB=/Library/NDI/lib/macOS
        fi
        export CPATH=${CPATH:+"$CPATH:"}/Library/NDI/include
        export DYLIBBUNDLER_FLAGS="${DYLIBBUNDLER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s $NDI_LIB"
        export LIBRARY_PATH=${LIBRARY_PATH:+"$LIBRARY_PATH:"}$NDI_LIB
        export MY_DYLD_LIBRARY_PATH="${MY_DYLD_LIBRARY_PATH:+$MY_DYLD_LIBRARY_PATH:}$NDI_LIB"
        echo "CPATH=$CPATH" >> $GITHUB_ENV
        echo "DYLIBBUNDLER_FLAGS=$DYLIBBUNDLER_FLAGS" >> $GITHUB_ENV
        echo "LIBRARY_PATH=$LIBRARY_PATH" >> $GITHUB_ENV
        echo "MY_DYLD_LIBRARY_PATH=$MY_DYLD_LIBRARY_PATH" >> $GITHUB_ENV
        cd $TEMP_INST
}

# Install live555
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles macosx
make -j $(sysctl -n hw.ncpu) install
cd ..

# Install Syphon
wget --no-verbose https://github.com/Syphon/Syphon-Framework/releases/download/5/Syphon.SDK.5.zip
unzip Syphon.SDK.5.zip
sudo cp -R 'Syphon SDK 5/Syphon.framework' /Library/Frameworks

install_ndi
install_ximea

# Install cross-platform deps
$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh

# Remove installation files
cd
rm -rf $TEMP_INST

