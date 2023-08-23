#!/bin/sh -eux

. /etc/os-release

if [ "$(id -u)" -eq 0 ]; then
        alias sudo=
fi

install_ximea() {
        curl -LO https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
        tar xzf XIMEA_Linux_SP.tgz
        cd package
        sudo ./install -noudev
}

# Install AJA
install_aja() {(
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

# Install live555
install_live555() {(
        git clone https://github.com/xanview/live555/
        cd live555
        git checkout 35c375
        ./genMakefiles linux-64bit
        make -j "$(nproc)" CPLUSPLUS_COMPILER="c++ -DXLOCALE_NOT_USED"
        sudo make install
)}

# Install NDI
install_ndi() {(
        if [ ! -f "Install_NDI_SDK_Linux.tar.gz" ]; then # it should be already cached in a CI step
                curl -L https://downloads.ndi.tv/SDK/NDI_SDK_Linux/Install_NDI_SDK_v5_Linux.tar.gz -o /var/tmp/Install_NDI_SDK_Linux.tar.gz
        fi
        tar -xzf Install_NDI_SDK_Linux.tar.gz
        # shellcheck disable=SC2125
        installer=./Install*NDI*sh
        yes | PAGER="cat" $installer
        sudo cp -r NDI\ SDK\ for\ Linux/include/* /usr/local/include
        sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' < 'NDI SDK for Linux/Version.txt' | sudo tee /usr/local/include/ndi_version.h
)}

# TODO: needed only for U20.04, remove after upgrading to U22.04
install_pipewire() {(
        if [ "$ID" = ubuntu ] && [ "$VERSION_ID" = 20.04 ]; then
                sudo apt install libdbus-1-dev meson
                git clone https://github.com/PipeWire/pipewire
                cd pipewire
                git checkout 19bcdaebe29b95edae2b285781dab1cc841be638 # last one supporting meson 0.53.2 in U20.04
                ./autogen.sh
                make -j "$(nproc)"
                sudo make install
        else
                sudo apt install libpipewire-0.3-dev
        fi
)}

# TODO: needed only for U18.04, remove after upgrading to U20.04
install_vulkan() {(
        git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers
        mkdir Vulkan-Headers/build
        cd Vulkan-Headers/build
        cmake ..
        sudo make install
        cd ../..
        git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Loader
        mkdir Vulkan-Loader/build
        cd Vulkan-Loader/build
        cmake ..
        cmake --build . --parallel
        sudo make install
)}

show_help=
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        show_help=1
fi

if [ $# -eq 0 ] || [ $show_help ]; then
        set -- aja gpujpeg live555 ndi pipewire vulkan ximea
fi

if [ $show_help ]; then
        set +x
        printf "Usage:\n"
        printf "\t%s [<features>] | [ -h | --help | help ]\n" "$0"
        printf "\nInstall all aditional dependencies (without arguments) or install one explicitly.\n"
        printf "\nAvailable ones: %s%s%s\n" "$(tput bold)" "$*" "$(tput sgr0)"
        return 0
fi

cd /var/tmp

while [ $# -gt 0 ]; do
        install_"$1"
        shift
done

