#!/bin/sh

git submodule update --init

./autogen.sh && rm Makefile

cd gpujpeg && ./autogen.sh && rm Makefile && cd ..

rm -rf .git .gitmodules
find -name .gitignore -print0 |xargs -0 rm
find -name .git -print0 |xargs -0 rm -rf

