#!/bin/bash -eux

# shellcheck disable=SC2140
printf "%b" "AJA_DIRECTORY=/var/tmp/ntv2\n"\
"CPATH=/usr/local/qt/include\n"\
"LIBRARY_PATH=/usr/local/qt/lib\n"\
"PKG_CONFIG_PATH=/usr/local/qt/lib/pkgconfig\n" >> "$GITHUB_ENV"
printf "/usr/local/qt/bin\n" >> "$GITHUB_PATH"

sudo add-apt-repository ppa:savoury1/vlc3 # new x265
sudo sed -n 'p; /^deb /s/^deb /deb-src /p' -i /etc/apt/sources.list # for build-dep ffmpeg
sudo apt update
sudo apt -y upgrade
sudo apt install appstream # appstreamcli for mkappimage AppStream validation
sudo apt install fonts-dejavu-core
sudo apt install libcppunit-dev
sudo apt --no-install-recommends install nvidia-cuda-toolkit
sudo apt install libglew-dev libglfw3-dev
sudo apt install libglm-dev
sudo apt install libx11-dev
sudo apt install libsdl2-dev libsdl2-ttf-dev
sudo apt install libspeexdsp-dev
sudo apt install libssl-dev
sudo apt install libasound-dev libjack-jackd2-dev libnatpmp-dev libv4l-dev portaudio19-dev

# updates nasm 2.13->2.14 in U18.04 (needed for rav1e)
update_nasm() {
        if [ -z "$(apt-cache search --names-only '^nasm-mozilla$')" ]; then
                return
        fi
        sudo apt install nasm- nasm-mozilla
        sudo ln -s /usr/lib/nasm-mozilla/bin/nasm /usr/bin/nasm
}

# for FFmpeg - libzmq3-dev needs to be ignored (cannot be installed, see run #380)
FFMPEG_BUILD_DEP=$(apt-cache showsrc ffmpeg | grep Build-Depends: | sed 's/Build-Depends://' | tr ',' '\n' |cut -f 2 -d\  | grep -v libzmq3-dev)
# shellcheck disable=SC2086
sudo apt install $FFMPEG_BUILD_DEP libdav1d-dev
sudo apt-get -y remove 'libavcodec*' 'libavutil*' 'libswscale*' libvpx-dev 'libx264*' nginx
update_nasm
sudo apt --no-install-recommends install asciidoc xmlto

sudo apt install libopencv-dev
sudo apt install libcurl4-nss-dev
sudo apt install i965-va-driver-shaders # instead of i965-va-driver
sudo apt install uuid-dev # Cineform

# Install cross-platform deps
"$GITHUB_WORKSPACE/.github/scripts/install-common-deps.sh"

sudo apt install qtbase5-dev
sudo chmod 777 /usr/local

install_ximea() {
        wget --no-verbose https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
        tar xzf XIMEA_Linux_SP.tgz
        cd package
        sudo ./install
}

# Install AJA
install_aja() {(
        cd /var/tmp
        git clone --depth 1 https://github.com/aja-video/ntv2
        cd ntv2/ajalibraries/ajantv2/build
        make -j "$(nproc)"
)}


install_gpujpeg() {(
        cd "$GITHUB_WORKSPACE"
        ./ext-deps/bootstrap_gpujpeg.sh -d
        mkdir ext-deps/gpujpeg/build
        cd ext-deps/gpujpeg/build
        cmake -DBUILD_OPENGL=OFF ..
        cmake --build . --parallel
        sudo cmake --install .
        sudo ldconfig
)}

# Install NDI
install_ndi() {
(
        cd /var/tmp
        [ -f Install_NDI_SDK_Linux.tar.gz ] || return 0
        tar -xzf Install_NDI_SDK_Linux.tar.gz
        # shellcheck disable=SC2125
        installer=./Install*NDI*sh
        yes | PAGER="cat" $installer
        sudo cp -r NDI\ SDK\ for\ Linux/include/* /usr/local/include
        sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' < 'NDI SDK for Linux/Version.txt' | sudo tee /usr/local/include/ndi_version.h
        sudo cp -r NDI\ SDK\ for\ Linux/lib/x86_64-linux-gnu/* /usr/local/lib
        sudo ldconfig
)
}

# Install live555
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles linux-64bit
make -j "$(nproc)" CPLUSPLUS_COMPILER="c++ -DXLOCALE_NOT_USED"
sudo make install
cd ..

install_aja
install_gpujpeg
install_ndi
install_ximea

