#!/bin/sh

if expr $GITHUB_REF : 'refs/heads/release/'; then
  VERSION=${GITHUB_REF#refs/heads/release/}
  TAG=v$VERSION
  CHANNEL=release
elif [ $GITHUB_REF = 'refs/heads/ndi-build' ]; then
  VERSION=ndi
  TAG=$VERSION
else
  VERSION=continuous
  TAG=$VERSION
fi

if [ -z ${CHANNEL-""} ]; then
        CHANNEL=$VERSION
fi

export CHANNEL TAG VERSION

echo "CHANNEL=$CHANNEL" >> $GITHUB_ENV
echo "TAG=$TAG" >> $GITHUB_ENV
echo "VERSION=$VERSION" >> $GITHUB_ENV
