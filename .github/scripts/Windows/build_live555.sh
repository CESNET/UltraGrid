#!/bin/bash -ex

export CC=gcc
export CXX=g++

cd /c
rm -rf live555
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles mingw
make -j $(nproc)

