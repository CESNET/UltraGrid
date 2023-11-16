#!/bin/sh -eux

## download directly release asset - doesn't work right now, the
## static library fails to link with the rest of the UG because
## the symbol __report_gsfailure in the lib (through MSVCRTD.lib)
## is also in in api-ms-win-crt-private-l1-1-0.dll
#install_cineform() {(
#        .github/scripts/download-gh-asset.sh gopro/cineform-sdk \
#                win-x64-release cineform.tar.gz
#        tar xaf cineform.tar.gz
#        cf_tgz=$PWD/cineform.tar.gz
#        cd /usr/local
#        tar xaf "$cf_tgz"
#        # fix "prefix=${prefix}" in .pc file
#        sed -i "1s-\${prefix}-/usr/local-" lib/pkgconfig/libcineformsdk.pc
#)}

build() {(
        cd /c
        rm -rf cineform-sdk
        git clone --depth 1 https://github.com/gopro/cineform-sdk
        c=cineform-sdk/build; mkdir $c; cd $c
        # workaround for needless Win GetProcessorCount definition
        sed -i 's/GetProcessorCount/_&/'  ../ConvertLib/ImageScaler.cpp
        cmake -DBUILD_STATIC=false -DBUILD_TOOLS=false -A x64 .. # assume
                                                  # "-G 'Visual Studio 16 2019'"
        cmake --build . --config Release --parallel
)}

install() {(
        mkdir -p /usr/local/bin /usr/local/include /usr/local/lib/pkgconfig
        cd /c/cineform-sdk/build
        cp Release/CFHDCodec.dll /usr/local/bin/
        cp Release/CFHDCodec.lib /usr/local/lib/
        cp ../Common/* /usr/local/include/
        cp libcineformsdk.pc /usr/local/lib/pkgconfig/
)}

$1
