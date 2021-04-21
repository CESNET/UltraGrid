#!/bin/sh -eu

# Use dylib bundler v2 from https://github.com/SCG82/macdylibbundler instead of the
# original because it has far better execution time (and perhaps also other improvements)

cd /tmp
git clone --depth 1 https://github.com/SCG82/macdylibbundler.git
cd macdylibbundler
make -j $(sysctl -n hw.ncpu)
sudo cp dylibbundler /usr/local/bin
cd -
rm -rf macdylibbundler

