#!/bin/sh -eux

curdir=$(cd "$(dirname "$0")"; pwd)
readonly curdir

win=no
case "$(uname -s)" in
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
                win=yes
                ;;

esac

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

download_install_cineform() {(
        git clone --depth 1 https://github.com/gopro/cineform-sdk
        cd cineform-sdk
        git apply "$curdir/0001-CMakeList.txt-remove-output-lib-name-force-UNIX.patch"
        cmake -DBUILD_STATIC=OFF -DBUILD_TOOLS=OFF -B build -S .
        cmake --build build --parallel "$(nproc)"
        sudo cmake --install build
)}

download_build_aja() {
        git clone --depth 1 https://github.com/aja-video/libajantv2.git
        # TODO TOREMOVE this workarounds when not needed
        tr -d '\n' < libajantv2/VERSION.txt > ver-fix-no-NL$$.txt &&
                mv ver-fix-no-NL$$.txt libajantv2/VERSION.txt
        sed -i -e '/MACOS_SDK_VERSION/d' libajantv2/cmake/CMakeOptions.cmake &&
                SDKROOT=$(xcrun --sdk macosx --show-sdk-path) && export SDKROOT
        export MACOSX_DEPLOYMENT_TARGET=10.13 # needed for arm64 mac

        cmake -DAJANTV2_DISABLE_DEMOS=ON  -DAJANTV2_DISABLE_DRIVER=ON \
                -DAJANTV2_DISABLE_TOOLS=ON  -DAJANTV2_DISABLE_TESTS=ON \
                -DAJANTV2_DISABLE_PLUGIN_LOAD=ON -DAJANTV2_BUILD_SHARED=ON \
                -DCMAKE_BUILD_TYPE=Release -Blibajantv2/build -Slibajantv2
        cmake --build libajantv2/build --config Release -j "$(nproc)"
}

install_aja() {(
        if [ "$(uname -s)" = Linux ]; then
                sudo apt install libudev-dev
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
)}

install_ews() {
        sudo mkdir -p /usr/local/include
        sudo curl -LSf https://raw.githubusercontent.com/hellerf/\
EmbeddableWebServer/master/EmbeddableWebServer.h -o \
/usr/local/include/EmbeddableWebServer.h
}

install_juice() {
(
        git clone https://github.com/paullouisageneau/libjuice.git
        mkdir libjuice/build
        cd libjuice/build
        cmake -DCMAKE_INSTALL_PREFIX=/usr/local -G "Unix Makefiles" ..
        make -j "$(nproc)"
        sudo make install
)
}

# fixes broken live555 test
live555_rm_tests() {
         sed -e '/TESTPROGS_DIR.*MAKE/d' Makefile > Makefile.fix
         mv -f Makefile.fix Makefile
}

download_build_live555() {(
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
)}

install_live555() {(
        if [ ! -d live555 ]; then
                download_build_live555
        fi
        sudo make -C live555 install
)}

install_pcp() {
        git clone https://github.com/libpcpnatpmp/libpcpnatpmp.git
        (
                cd libpcpnatpmp
                # TODO TOREMOVE when not needed
                if is_win; then
                        git checkout 46341d6
                        sed "/int gettimeofday/i\\
struct timezone;\\
struct timeval;\\
" libpcp/src/windows/pcp_gettimeofday.h > fixed
                        mv fixed libpcp/src/windows/pcp_gettimeofday.h
                fi
                sed 's/AC_PREREQ(.*)/AC_PREREQ(\[2.69\])/' configure.ac \
                        > configure.ac.fixed
                mv configure.ac.fixed configure.ac

                ./autogen.sh || true # autogen exits with 1
                CFLAGS=-fPIC ./configure --disable-shared
                make -j "$(nproc)"
                sudo make install
        )
        rm -rf libpcpnatpmp
}

install_zfec() {(
        git clone --depth 1 https://github.com/tahoe-lafs/zfec zfec
        sudo mkdir -p /usr/local/src
        sudo mv zfec/zfec /usr/local/src
)}

if ! is_arm && ! is_win; then
        download_install_cineform
fi
install_aja
install_ews
install_juice
install_live555
install_pcp
install_zfec

