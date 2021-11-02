#!/bin/sh

NDI=--disable-ndi

if expr $GITHUB_REF : 'refs/tags/'; then
  TAG=${GITHUB_REF#refs/tags/}
  VERSION=${TAG#v}
  CHANNEL=release
elif [ $GITHUB_REF = 'refs/heads/ndi-build' ]; then
  NDI=--enable-ndi
  VERSION=ndi
  TAG=$VERSION
else
  VERSION=continuous
  TAG=$VERSION
fi

if [ -z ${CHANNEL-""} ]; then
        CHANNEL=$VERSION
fi

export CHANNEL NDI TAG VERSION

echo "CHANNEL=$CHANNEL" >> $GITHUB_ENV
echo "NDI=$NDI" >> $GITHUB_ENV
echo "TAG=$TAG" >> $GITHUB_ENV
echo "VERSION=$VERSION" >> $GITHUB_ENV
