#!/bin/sh -eux

case "$(uname -s)" in
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
                ;;

        *)
                sudo="sudo"
                ;;
esac

if ! command -v nproc >/dev/null; then
        nproc() { sysctl -n hw.logicalcpu; } # mac
fi

# only download here, compilation is handled per-platform
download_cineform() {(
        cd "$GITHUB_WORKSPACE"
        git clone --depth 1 https://github.com/gopro/cineform-sdk
        mkdir cineform-sdk/build
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

download_cineform
install_ews
install_juice
install_pcp
install_zfec

