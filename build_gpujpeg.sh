#!/bin/bash

PWD=`pwd`

cd gpujpeg

export MSVC_PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio\ 12.0/
export PATH=$PATH:$MSVC_PATH/Common7/IDE/:$MSVC_PATH/VC/bin/
export PATH=$PATH:$CUDA_PATH\\bin
export INCLUDE=.

nvcc -DGPUJPEG_EXPORTS -o gpujpeg.dll --shared src/gpujpeg_*c src/gpujpeg*cu

cp gpujpeg.lib /usr/local/lib
cp gpujpeg.dll /usr/local/bin
cp -r libgpujpeg /usr/local/include

cd $PWD

