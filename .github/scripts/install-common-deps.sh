#!/bin/sh -eux

case "$(uname -s)" in
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
                SUDO=
                ;;

        *)
                SUDO="sudo "
                ;;
esac

install_pcp() {
        git clone https://github.com/MartinPulec/pcp.git
        (
                cd pcp
                ./autogen.sh || true # autogen exits with 1
                CFLAGS=-fPIC ./configure --disable-shared
                make -j 5
                ${SUDO}make install
        )
        rm -rf pcp
}

install_zfec() {
        ( cd $GITHUB_WORKSPACE && git submodule update --init ext-deps/zfec || exit 1 )
}

install_pcp
install_zfec

