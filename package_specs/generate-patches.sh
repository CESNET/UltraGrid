#!/bin/bash

scriptroot="$(dirname "$(realpath "$0")")"

vendors=( deltacast blackmagick dvs aja bluefish cuda )
subpackages=( ultragrid-nightly ultragrid ultragrid-proprietary-drivers )

mkdir a
for pkg in ${subpackages[*]} ; do
	cp -r $pkg a/
	if test -f "a/$pkg/debian.tar.gz" ; then
		tar -C a/$pkg -xf "a/$pkg/debian.tar.gz"
	fi 
done

for vendor in ${vendors[*]} ; do
	cp -r a b.$vendor
	
	for pkg in ${subpackages[*]} ; do
		FILEMASK="a/$pkg/*.spec a/$pkg/*.spec.tpl a/$pkg/*.dsc.tpl a/$pkg/*.dsc"
		if test -f "a/$pkg/debian/rules" ; then
			FILEMASK+=" a/$pkg/debian/rules"
		fi
		if test -f "a/$pkg/debian.rules" ; then
			FILEMASK+=" a/$pkg/debian.rules"
		fi
		for specfile in $(echo $FILEMASK ) ; do
			if test -f "$specfile" ; then
				cat $specfile | ${scriptroot}/comment.py on "$vendor" > "$(echo $specfile | sed -r "s#^a/#b.$vendor/#g")"
			fi
		done
	done

	diff -rupN a b.$vendor | sed "s#b.$vendor/#b/#g" > __disable_${vendor}__.patch

	rm -r b.$vendor
done

rm -r a


