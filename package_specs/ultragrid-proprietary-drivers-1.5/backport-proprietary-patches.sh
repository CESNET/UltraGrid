#!/bin/bash

function err_exit(){
	printf "%s\n" "$2"
	exit $1
}

function usage(){
	printf "%s drivers.tar 4.9.2 opensuse-422\n" "$0"
	printf "\tdrivers.tar\tpath to tarball with proprietary code\n"
	printf "\t4.9.2\t\ttarget kernel version to backport patches onto\n"
	printf "\topensuse-422\tAppend tag to patch name\n"
}

function backport_patch(){
	local doi="$4"
	local tag=$3
	local kver=$2
	local inpatch="$1"
	local outpatch="$(echo "$inpatch" | sed -r 's#[.]patch$#-$tag.patch#g')"
	# patches on input are sorted, b was reset before call

	local kver_MAJOR="$(echo "$kver" | cut -d'.' -f1)"
	local kver_MINOR="$(echo "$kver" | cut -d'.' -f2)"
	local kver_RELEASE="$(echo "$kver" | cut -d'.' -f3)"
	
	#[[ "x$inpatch" != "xbackport" ]] && patch -p1 <"$inpatch"

	# scan loop
	local matchver='KERNEL_VERSION[[:space:]]*[(][[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*[)]'
	local matchexpr='#if.*\bLINUX_VERSION_CODE\b.*\b('"$matchver"').*'
	
	grep -HorE "$matchexpr" "$doi" | dos2unix | while read -d$'\n' -r KVER ; do
		local KVER_BASE="$(echo "$KVER" | cut -d: -f2)"
		local KVER_MAJOR="$(echo "$KVER_BASE" | sed -r "s#^.*$matchver.*\$#\1#g")"
		local KVER_MINOR="$(echo "$KVER_BASE" | sed -r "s#^.*$matchver.*\$#\2#g")"
		local KVER_RELEASE="$(echo "$KVER_BASE" | sed -r "s#^.*$matchver.*\$#\3#g")"

		[[ "$KVER_MAJOR" -lt "$kver_MAJOR" ]] && continue
		([[ "$KVER_MAJOR" = "$kver_MAJOR" ]] && [[ "$KVER_MINOR" -lt "$kver_MINOR" ]]) && continue
		([[ "$KVER_MAJOR" = "$kver_MAJOR" ]] && [[ "$KVER_MINOR" = "$kver_MINOR" ]] && [[ "$KVER_RELEASE" -lt "$kver_RELEASE" ]]) && continue

		sed -i '-e' "s#$(echo "$KVER" | grep -oE "$matchver")#KERNEL_VERSION($kver_MAJOR,$kver_MINOR,$kver_RELEASE)#g" "$(echo "$KVER" | cut -d':' -f1)"
	done

	# end of replace, generate patch
}

scriptroot="$(dirname "$(realpath "$0")")"
drivers="$1"
tokernel="$2"
patchtag="$3"

for arg; do
	[[ "x$arg" = "x-h" ]] && (usage "$0" ; exit 1)
done

[[ "x$drivers" = "x" ]] && err_exit 1 "Need tarball with proprietary drivers to extract"
[[ "x$tokernel" = "x" ]] && err_exit 1 "Need target kernel version"
[[ "x$patchtag" = "x" ]] && err_exit 1 "Need patch tagname"

vendors=( deltacast blackmagick dvs AJA bluefish cuda )
subdirs=( /dev/null '*ideoMaster*' 'sdk*' 'ntv2sdk*' 'Epoch*' /dev/null )

mkdir -p backport_tmp
rm -rf a

echo -n "extracting drivers ..."
tar -C backport_tmp -xf "$drivers"
echo " [done]"

mv backport_tmp/ultragrid-proprietary-drivers* a
rm -rf backport_tmp

for i in $(seq 0 $((${#vendors[@]}-1))) ; do
	vendor=${vendors[$i]}
	vendir="${subdirs[$i]}" 
	[[ "x$vendir" = "x/dev/null" ]] && continue
	vendir="$(echo "a/"$vendir)"

	echo "processing $vendor dir $vendir"
	[[ -d "$vendir" ]] || continue

	cp -r a a.$vendor

	patchconv=$((ls -1 "../$vendor"-linux[0-9]*.patch "../${subdirs[$i]}"-linux[0-9]*.patch 2>/dev/null | sort -n | head -n 1 | cut -d- -f1 ; echo "$vendor" ) | head -n 1)

	pushd a.$vendor > /dev/null
	for patch in $(ls -1 "../$vendor"-linux[0-9]*.patch "../${subdirs[$i]}"-linux[0-9]*.patch |& sort -n) ; do
		[[ ! -f "$patch" ]] && continue
		patch -p1 <"$patch"
	done
	popd > /dev/null
	printf "\tpatches applied (if any)\n"


	cp -r a.$vendor b.$vendor
	pushd b.$vendor > /dev/null
	backport_patch "backport" "$tokernel" "$patchtag" "$(basename "$vendir")"
	popd > /dev/null

	outpatch="$patchconv-kernel-backports-$patchtag.patch"
	
	diff -rupN a.$vendor b.$vendor > "$outpatch"
	sed -i -r -e "s#\bb.$vendor\b/#b/#g" -e "s#\ba.$vendor\b/#a/#g" "$outpatch" 
	[[ -s "$outpatch" ]] || rm "$outpatch"
	

	rm -r b.$vendor
	rm -r a.$vendor
done

rm -r a

