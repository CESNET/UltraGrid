#!/bin/bash -ex

cd /c
git clone https://github.com/xanview/live555/
cd live555
git checkout 35c375
./genMakefiles mingw
make -j $(nproc)

