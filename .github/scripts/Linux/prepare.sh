#!/bin/bash -eux

echo "AJA_DIRECTORY=/var/tmp/ntv2sdk" >> $GITHUB_ENV
echo "CPATH=/usr/local/qt/include" >> $GITHUB_ENV
echo "LIBRARY_PATH=/usr/local/qt/lib" >> $GITHUB_ENV
echo "PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig" >> $GITHUB_ENV
echo "/usr/local/qt/bin" >> $GITHUB_PATH

if command -v gcc-5; then
        CUDA_HOST_COMPILER=gcc-5
else
        CUDA_HOST_COMPILER=gcc-6
fi
echo "CUDA_HOST_COMPILER=$CUDA_HOST_COMPILER" >> $GITHUB_ENV

sudo sed -n 'p; /^deb /s/^deb /deb-src /p' -i /etc/apt/sources.list # for build-dep ffmpeg
sudo apt update
sudo apt install libcppunit-dev nvidia-cuda-toolkit
sudo apt install libglew-dev freeglut3-dev libgl1-mesa-dev
sudo apt install libx11-dev
sudo apt install libsdl2-dev
sudo apt install libssl-dev
sudo apt install portaudio19-dev libjack-jackd2-dev libasound-dev libv4l-dev

# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
FFMPEG_BUILD_DEP=`apt-cache showsrc ffmpeg | grep Build-Depends: | sed 's/Build-Depends://' | tr ',' '\n' |cut -f 2 -d\  | grep -v libzmq3-dev`
sudo apt install $FFMPEG_BUILD_DEP
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' 'libx264*' nasm
sudo apt --no-install-recommends install asciidoc xmlto

sudo apt install libopencv-dev
sudo apt install libglib2.0-dev libcurl4-nss-dev
sudo apt install libtool # gpujpeg
( ./bootstrap_gpujpeg.sh -d && mkdir ext-deps/gpujpeg/build && cd ext-deps/gpujpeg/build && CUDA_FLAGS=-D_FORCE_INLINES CXXFLAGS=-std=c++11 CC=$CUDA_HOST_COMPILER ../autogen.sh && make && sudo make install && sudo ldconfig || exit 1 )
( sudo apt install uuid-dev && git submodule update --init cineform-sdk && cd cineform-sdk/ && cmake -DBUILD_TOOLS=OFF . && make CFHDCodecStatic )
sudo apt install qtbase5-dev
sudo chmod 777 /usr/local

# Install XIMEA
wget --no-verbose https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
tar xzf XIMEA_Linux_SP.tgz
cd package
sudo ./install

# Install AJA
if [ -n "$SDK_URL" ]; then
        curl -S $SDK_URL/ntv2sdklinux.zip -O
        unzip ntv2sdklinux.zip -d /var/tmp
        mv /var/tmp/ntv2sdk* /var/tmp/ntv2sdk
        cd /var/tmp/ntv2sdk/ajalibraries/ajantv2
        export CXX='g++ -std=gnu++11'
        make
fi

# Install NDI
if [ -n "$SDK_URL" -a "$GITHUB_REF" = refs/heads/ndi-build ]; then
        curl -S $SDK_URL/NDISDK_Linux.tar.gz -O
        tar -C /var/tmp -xzf NDISDK_Linux.tar.gz
        yes | PAGER=cat /var/tmp/Install*NDI*sh
	sudo cp -r NDI\ SDK\ for\ Linux/include/* /usr/local/include
	sudo cp -r NDI\ SDK\ for\ Linux/lib/x86_64-linux-gnu/* /usr/local/lib
	sudo ldconfig
fi

# Install live555
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles linux-64bit
sudo make install CPLUSPLUS_COMPILER="c++ -DXLOCALE_NOT_USED"

cd ..

