#!/bin/sh -eux

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

