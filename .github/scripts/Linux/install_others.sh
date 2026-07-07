#!/bin/sh -eux

. /etc/os-release

if [ "$(id -u)" -eq 0 ]; then
        alias sudo=
fi

install_ximea() (
        filename=XIMEA.tgz
        if [ ! -f "$filename" ]; then
                curl -L "${XIMEA_DOWNLOAD_URL:?}" -o $filename
        fi
        tar xzf $filename
        cd package
        sudo ./install -noudev
)

install_gpujpeg() (
        curl -LO https://github.com/CESNET/GPUJPEG/releases/download/\
continuous/GPUJPEG-Linux.tar.xz
        tar xaf GPUJPEG-Linux.tar.xz
        sudo cp -r GPUJPEG/* /usr/local/
        sudo ldconfig
)

# Install NDI
install_ndi() (
        if [ ! -f Install_NDI_SDK_Linux.tar.gz ]; then
                curl -Lf https://downloads.ndi.tv/SDK/NDI_SDK_Linux/\
Install_NDI_SDK_v6_Linux.tar.gz -o /var/tmp/Install_NDI_SDK_Linux.tar.gz
        fi
        tar -xzf Install_NDI_SDK_Linux.tar.gz
        # shellcheck disable=SC2125
        installer=./Install*NDI*sh
        yes | PAGER="cat" $installer
        sudo cp -r NDI\ SDK\ for\ Linux/include/* /usr/local/include/
)

install_svt_jpegxs() (
        git clone --depth 1 https://github.com/OpenVisualCloud/SVT-JPEG-XS
        # when built in U22.04, the stack is set as executable for some reason,
        # not in Arch, eg.
        export LDFLAGS="-Wl,-z,noexecstack"
        cmake -B SVT-JPEG-XS/build SVT-JPEG-XS
        cmake --build SVT-JPEG-XS/build --parallel "$(nproc)"
        sudo cmake --install SVT-JPEG-XS/build
)

# FFmpeg master needs at least v1.3.277 as for 6th Mar '25
install_vulkan() (
        sudo apt build-dep libvulkan1
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
)

show_help=
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        show_help=1
fi

if [ $# -eq 0 ] || [ $show_help ]; then
        set -- gpujpeg ndi svt_jpegxs vulkan ximea
fi

if [ $show_help ]; then
        set +x
        printf "Usage:\n"
        printf "\t%s [<features>] | [ -h | --help | help ]\n" "$0"
        printf "\nInstall all aditional dependencies (without arguments) or install one explicitly.\n"
        printf "\nAvailable ones: %s%s%s\n" "$(tput bold)" "$*" "$(tput sgr0)"
        exit 0
fi

cd /var/tmp

while [ $# -gt 0 ]; do
        install_"$1"
        shift
done

