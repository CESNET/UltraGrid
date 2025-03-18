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
        mkdir build && cd build
        cmake -DBUILD_TOOLS=OFF ..
        cmake --build . --parallel "$(nproc)"
        sudo cmake --install .
)}

install_ews() {
        sudo mkdir -p /usr/local/include
        sudo curl -LS https://raw.githubusercontent.com/hellerf/\
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

install_pcp() {
        git clone https://github.com/libpcp/pcp.git
        (
                cd pcp
                # TODO TOREMOVE when not needed
                sed "/int gettimeofday/i\\
struct timezone;\\
struct timeval;\\
" libpcp/src/windows/pcp_gettimeofday.h > fixed
                mv fixed libpcp/src/windows/pcp_gettimeofday.h

                ./autogen.sh || true # autogen exits with 1
                CFLAGS=-fPIC ./configure --disable-shared
                make -j "$(nproc)"
                sudo make install
        )
        rm -rf pcp
}

install_zfec() {(
        git clone --depth 1 https://github.com/tahoe-lafs/zfec zfec
        sudo mkdir -p /usr/local/src
        sudo mv zfec/zfec /usr/local/src
)}

if ! is_arm && ! is_win; then
        download_install_cineform
fi
install_ews
install_juice
install_pcp
install_zfec

