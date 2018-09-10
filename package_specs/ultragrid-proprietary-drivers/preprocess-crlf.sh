#!/bin/bash

tar -xvf drivers.tar.gz || exit 1
drvdir=$(find . -maxdepth 1 -type d -iname 'ultragrid-proprietary-drivers-*')
pushd $drvdir || exit 2

for filepattern in Makefile ; do
	find . -iname "$filepattern" -exec dos2unix {} \;
done
popd
tar -c $drvdir -zf drivers.new.tar.gz || exit 3
mv drivers.new.tar.gz drivers.tar.gz
