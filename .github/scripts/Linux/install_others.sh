#!/bin/sh -eux

. /etc/os-release

if [ "$(id -u)" -eq 0 ]; then
        alias sudo=
fi

install_ximea() {
        if [ "$(dpkg --print-architecture | cut -b1-3)" = arm ]; then
                filename=XIMEA_Linux_ARM_SP.tgz
        else
                filename=XIMEA_Linux_SP.tgz
        fi
        curl -LO https://www.ximea.com/downloads/recent/$filename
        tar xzf $filename
        cd package
        touch bin/streamViewer.64 # TOREMOVE
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
        cmake --build . --parallel "$(nproc)"
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
        tar -xzf Install_NDI_SDK_Linux.tar.gz
        # shellcheck disable=SC2125
        installer=./Install*NDI*sh
        yes | PAGER="cat" $installer
        sudo cp -r NDI\ SDK\ for\ Linux/include/* /usr/local/include/
        sed 's/\(.*\)/\#define NDI_VERSION \"\1\"/' < 'NDI SDK for Linux/Version.txt' | sudo tee /usr/local/include/ndi_version.h
)}

# TODO: needed only for U20.04, remove after upgrading to U22.04
install_pipewire() {(
        if { [ "$ID" = ubuntu ] && [ "$VERSION_ID" = 20.04 ]; } ||
                { [ "${ID_LIKE-$ID}" = debian ] && [ "$VERSION_ID" -le 11 ]; }
        then
                sudo apt -y install libdbus-1-dev meson
                git clone https://github.com/PipeWire/pipewire
                cd pipewire
                git checkout 19bcdaebe29b95edae2b285781dab1cc841be638 # last one supporting meson 0.53.2 in U20.04
                ./autogen.sh -Dtests=disabled
                make -j "$(nproc)"
                sudo make install
        else
                sudo apt -y install libpipewire-0.3-dev
        fi
)}

install_rav1e() {(
        # TODO: use avx2 later
        if expr "${UG_ARCH-}" : '.*avx' >/dev/null; then
                avx2=avx2
        fi
        fpattern="librav1e.*linux-${avx2-sse4}.tar.gz"
        "${GITHUB_WORKSPACE-.}"/.github/scripts/download-gh-asset.sh xiph/rav1e \
                "$fpattern" librav1e.tar.gz
        sudo tar xaf librav1e.tar.gz -C /usr/local
        sudo rm -rf /usr/local/lib/librav1e.so*
        sudo sed -i -e 's-prefix=dist-prefix=/usr/local-' \
                -e 's/-lrav1e/-lrav1e -lm -pthread/' \
                /usr/local/lib/pkgconfig/rav1e.pc
)}

# FFmpeg master needs at least v1.3.238 as for 23th Aug '23
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
        cmake --build . --parallel "$(nproc)"
        sudo make install
)}

show_help=
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        show_help=1
fi

if [ $# -eq 0 ] || [ $show_help ]; then
        set -- aja gpujpeg live555 ndi pipewire rav1e vulkan ximea
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

