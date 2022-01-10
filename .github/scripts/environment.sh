#!/bin/sh

if expr $GITHUB_REF : 'refs/tags/'; then
  TAG=${GITHUB_REF#refs/tags/}
  VERSION=${TAG#v}
  CHANNEL=release
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
