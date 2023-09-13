#!/bin/sh -eux

curdir=$(cd "$(dirname "$0")"; pwd)
readonly curdir

win=no
case "$(uname -s)" in
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
                win=yes
                ;;

        *)
                if [ "$(id -u)" -ne 0 ]; then
                        sudo="sudo"
                fi
                ;;
esac

if ! command -v nproc >/dev/null; then
        nproc() { sysctl -n hw.logicalcpu; } # mac
fi

is_arm() { expr "$(dpkg --print-architecture)" : arm >/dev/null; }

# for Win only download here, compilation is handled differently
download_install_cineform() {(
        git clone --depth 1 https://github.com/gopro/cineform-sdk
        cd cineform-sdk
        git apply "$curdir/0001-CMakeList.txt-remove-output-lib-name-force-UNIX.patch"
        mkdir build && cd build
        if [ "$win" = no ]; then
                cmake -DBUILD_TOOLS=OFF ..
                cmake --build . --parallel
                sudo cmake --install .
        fi
)}

install_ews() {
        ${sudo:+"$sudo" }curl -LS https://raw.githubusercontent.com/hellerf/EmbeddableWebServer/master/EmbeddableWebServer.h -o /usr/local/include/EmbeddableWebServer.h
}

install_juice() {
(
        git clone https://github.com/paullouisageneau/libjuice.git
        mkdir libjuice/build
        cd libjuice/build
        cmake -DCMAKE_INSTALL_PREFIX=/usr/local -G "Unix Makefiles" ..
        make -j "$(nproc)"
        ${sudo:+"$sudo" }make install
)
}

install_pcp() {
        git clone https://github.com/libpcp/pcp.git
        (
                cd pcp
                ./autogen.sh || true # autogen exits with 1
                CFLAGS=-fPIC ./configure --disable-shared
                make -j "$(nproc)"
                ${sudo+"$sudo" }make install
        )
        rm -rf pcp
}

install_zfec() {(
        git clone --depth 1 https://github.com/tahoe-lafs/zfec zfec
        ${sudo:+"$sudo" }mkdir -p /usr/local/src
        ${sudo:+"$sudo" }mv zfec/zfec /usr/local/src
)}

if ! is_arm; then
        download_install_cineform
fi
install_ews
install_juice
install_pcp
install_zfec

