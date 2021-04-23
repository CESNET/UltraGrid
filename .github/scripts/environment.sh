#!/bin/sh

NDI=--disable-ndi

if expr $GITHUB_REF : 'refs/heads/release/'; then
  VERSION=${GITHUB_REF#refs/heads/release/}
  TAG=v$VERSION
elif [ $GITHUB_REF = 'refs/heads/ndi-build' ]; then
  NDI=--enable-ndi
  VERSION=ndi
  TAG=$VERSION
else
  VERSION=continuous
  TAG=$VERSION
fi

export NDI TAG VERSION

echo "NDI=$NDI" >> $GITHUB_ENV
echo "TAG=$TAG" >> $GITHUB_ENV
echo "VERSION=$VERSION" >> $GITHUB_ENV
