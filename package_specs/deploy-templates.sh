#!/bin/bash

DIR="$(dirname "$0")"
for specfile in $(find "$DIR" -name "*.spec.tpl" -or -name "*.dsc.tpl") ; do
	cp "$specfile" $(echo $specfile | sed -r 's#.tpl$##g')
done
