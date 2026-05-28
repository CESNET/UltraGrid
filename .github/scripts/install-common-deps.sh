#!/bin/sh -eu

curdir=$(cd "$(dirname "$0")"; pwd)
readonly curdir

win=no
case "$(uname -s)" in
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
                win=yes
                ;;
esac

if [ -f /etc/os-release ]; then
        . /etc/os-release
else
        ID=
fi

if ! command -v nproc >/dev/null; then
        nproc() { sysctl -n hw.logicalcpu; } # mac
fi

is_arm() {
        uname_m=$(uname -m)
        expr "$uname_m" : arm > /dev/null || [ "$uname_m" = aarch64 ]
}
is_win() { [ "$win" = yes ]; }

if is_win || [ "$(id -u)" -eq 0 ]; then
        alias sudo=
fi

install_cineform() (
        git clone --depth 1 https://github.com/gopro/cineform-sdk
        cd cineform-sdk
        git apply "$curdir"/patches/\
cineform-0001-CMakeList.txt-remove-output-lib-name-force-UNIX.patch
        cmake -DBUILD_STATIC=OFF -DBUILD_TOOLS=OFF -B build -S .
        cmake --build build --parallel "$(nproc)"
        sudo cmake --install build
)

download_build_aja() (
        aja_url=https://github.com/aja-video/libajantv2.git
        git clone --depth 1 $aja_url
        cd libajantv2
        latest_tag=$(git ls-remote --tags origin | awk '{print $2}' | \
                grep 'refs/tags/ntv2_[0-9]' | grep -Ev 'beta|rc' | sort | \
                tail -n 1)
        git fetch --depth=1 origin "$latest_tag"
        git checkout FETCH_HEAD
        cmake -DAJANTV2_DISABLE_DEMOS=ON  -DAJANTV2_DISABLE_DRIVER=ON \
                -DAJANTV2_DISABLE_TOOLS=ON  -DAJANTV2_DISABLE_TESTS=ON \
                -DAJANTV2_DISABLE_PLUGIN_LOAD=ON -DAJANTV2_BUILD_SHARED=ON \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=macosx \
                -Bbuild -S.
        cmake --build build --config Release -j "$(nproc)"
)

install_aja() (
        if [ "$ID" = ubuntu ]; then
                sudo apt install libudev-dev
        elif [ "$ID" = almalinux ]; then
                sudo dnf -y install systemd-devel
        fi
        if [ ! -d libajantv2 ]; then
                download_build_aja
        fi
        if is_win; then
                cd libajantv2/build/ajantv2/Release
                cp ajantv2*.dll /usr/local/bin/
                cp ajantv2*.lib /usr/local/lib/
        else
                sudo cmake --install libajantv2/build
        fi
)

install_ews() {
        sudo mkdir -p /usr/local/include
        sudo curl -LSf https://raw.githubusercontent.com/hellerf/\
EmbeddableWebServer/master/EmbeddableWebServer.h -o \
/usr/local/include/EmbeddableWebServer.h
}

install_juice() (
        git clone https://github.com/paullouisageneau/libjuice.git
        mkdir libjuice/build
        cd libjuice/build
        cmake -DCMAKE_INSTALL_PREFIX=/usr/local -G "Unix Makefiles" ..
        make -j "$(nproc)"
        sudo make install
)

# fixes broken live555 test
live555_rm_tests() {
         sed -e '/TESTPROGS_DIR.*MAKE/d' Makefile > Makefile.fix
         mv -f Makefile.fix Makefile
}

download_build_live555() (
        git clone --depth 1 https://github.com/xanview/live555/
        cd live555

        if is_win; then
                ./genMakefiles mingw
                PATH=/usr/bin:$PATH
                # ensure binutils ld is used (not lld)
                pacman -Sy --noconfirm binutils
                make -j "$(nproc)" CXX="c++ -DNO_GETIFADDRS -DNO_OPENSSL"
                pacman -Rs --noconfirm binutils
        elif [ "$(uname -s)" = Linux ]; then
                ./genMakefiles linux-with-shared-libraries
                make -j "$(nproc)" CPLUSPLUS_COMPILER="c++ -DNO_STD_LIB"
        else
                ./genMakefiles macosx-no-openssl
                live555_rm_tests
                make -j "$(nproc)" CPLUSPLUS_COMPILER="c++ -std=c++11"
        fi
)

install_live555() (
        if [ ! -d live555 ]; then
                download_build_live555
        fi
        sudo make -C live555 install
)

install_omt() (
        if [ "$(uname -s)" = Darwin ]; then
                # brew install dotnet-sdk # already installed in CI
                build=buildmacuniversal.sh
                if is_arm; then omtdir=osx-arm64; else omtdir=osx-x64; fi
                libext=dylib
                printf "MACOS_BUNDLE_EXTRA_LIBS=%s /usr/local/lib/libvmx.dylib\n"\
                        "${MACOS_BUNDLE_EXTRA_LIBS-}" >> "$GITHUB_ENV"
        else # Linux
                # if [ "$(uname -m)" = aarch64 ]; then
                #         # https://learn.microsoft.com/en-us/dotnet/core/install/linux-scripted-manual#scripted-install
                #         curl -LO https://dot.net/v1/dotnet-install.sh
                #         chmod +x dotnet-install.sh
                #         ./dotnet-install.sh --channel 8.0
                #         sudo apt -y install clang zlib1g-dev
                #         # shellcheck disable=SC2031 # no problem
                #         PATH=$PATH:$HOME/.dotnet
                #         build=buildlinuxarm64.sh
                #         omtdir=linux-arm64
                if [ "$ID" != ubuntu ]; then
                        return # no build for Debian (ARM) and AlmaLinux (alt)
                fi
                sudo apt install dotnet8
                build=buildlinuxx64.sh
                omtdir=linux-x64
                libext=so
        fi

        mkdir omt_build
        cd omt_build
        git clone --depth 1 https://github.com/openmediatransport/libvmx.git
        git clone --depth 1 https://github.com/openmediatransport/libomt.git
        git clone --depth 1 https://github.com/openmediatransport/libomtnet.git

        cd libvmx/build
        chmod +x $build
        ./$build
        cd ../..

        cd libomtnet/build
        chmod +x buildall.sh
        ./buildall.sh
        cd ../..

        cd libomt/build
        chmod +x $build
        ./$build
        cd ../..

        sudo cp libvmx/build/libvmx.$libext /usr/local/lib/
        sudo cp libomt/bin/Release/net8.0/$omtdir/publish/libomt.h /usr/local/include/
        sudo cp libomt/bin/Release/net8.0/$omtdir/publish/libomt.$libext /usr/local/lib/
)

install_pcp() {
        git clone https://github.com/libpcpnatpmp/libpcpnatpmp.git
        (
                cd libpcpnatpmp
                sed 's/AC_PREREQ(.*)/AC_PREREQ(\[2.69\])/' configure.ac \
                        > configure.ac.fixed
                mv configure.ac.fixed configure.ac
                ./autogen.sh
                ./configure
                make -j "$(nproc)"
                sudo make install
        )
        rm -rf libpcpnatpmp
}

install_zfec() (
        git clone --depth 1 https://github.com/tahoe-lafs/zfec zfec
        sudo mkdir -p /usr/local/src
        sudo mv zfec/zfec /usr/local/src
)

install_items="aja ews juice live555 pcp zfec"
if ! is_arm && ! is_win; then
        install_items="$install_items cineform"
fi
if ! is_win; then
        install_items="$install_items omt"
fi

if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] || [ "$1" = help ]; }; then
        printf "Usage:\n"
        printf "\t%s [<features>] | [ -h | --help | help ]\n" "$0"
        printf "\nInstall all aditional dependencies (without arguments) or \
install one explicitly.\n"
        printf "\nAvailable features: %s%s%s\n" "$(tput bold)" "$install_items" "$(tput sgr0)"
        exit 0
fi

if [ $# -eq 0 ]; then
        # shellcheck disable=SC2086 # intentional
        set -- $install_items
fi

set -x

while [ $# -gt 0 ]; do
        install_"$1"
        shift
done
