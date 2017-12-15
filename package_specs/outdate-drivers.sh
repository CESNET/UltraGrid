#!/bin/bash

function patch_file(){
	local filename="$2"
	local entry="$1"
	local lockplace="# replace this line with generated conflicts"
	grep -vE '^((Provides)|(Conflicts))[:]?[[:space:]]*ultragrid-proprietary-drivers.*' "$filename" > "$filename.$$"
	sed -i "s/$lockplace.*\$/$entry\n$lockplace/g" "$filename.$$"
	mv "$filename.$$" "$filename"
}
function patch_pkgname(){
	local filename="$1"
	local entry="$2"
	sed -ri "/^Source:.*\$/b; s/ultragrid-proprietary-drivers((-rpmlintrc)?)\$/ultragrid-proprietary-drivers-$entry\1/g" "$filename"
	sed -ri "s/^ultragrid-proprietary-drivers\b/ultragrid-proprietary-drivers-$entry/g" "$filename"
}
function move_file(){
	local lintfile="$1"
	local lockver="$2"
	mv "$lintfile" "$(echo "$lintfile" | sed "s#ultragrid-proprietary-drivers#ultragrid-proprietary-drivers-$lockver#g")"
}
function debdirpatch(){
	local prefix="$1"
	local archivefile="$2"
	local lockver="$3"
	local conflicttext="$4"
	local providetext="$5"

	if [[ "x$archivefile" != "x" ]] && [[ -f "$archivefile" ]] ; then
		rm -rf debian
		tar -xf "$debarchive"
	fi

	if [[ -f "${prefix}control" ]] ; then
		patch_file "$conflicttext\n$providetext" "${prefix}control"
		patch_pkgname "${prefix}control" "$lockver"
	fi
	
	local installfile="${prefix}ultragrid-proprietary-drivers.install"
	if [[ -f "$installfile" ]] ; then
		move_file "$installfile" "$lockver"
	fi

	for lintfile in "${prefix}"*lintian* "${prefix}"source/*lintian*  ; do 
		test -f "$lintfile" || continue
		patch_pkgname "$lintfile" "$lockver"
	done

	if [[ "x$archivefile" != "x" ]] && [[ -f "$archivefile" ]] ; then
		tar -c debian -zf "$debarchive"
		rm -rf debian
	fi
}
function err_exit(){
	printf "%s\n" "$1" >&2
	exit $2
}

scriptroot="$(dirname "$(realpath "$0")")"

driverssubdir=ultragrid-proprietary-drivers
lockver="$1"
newdir="$scriptroot/$driverssubdir-$lockver"

[[ "x$lockver" != "x" ]] || err_exit "usage: $0 <version to branch> <space separated conflicting versions>" 1
[[ -d "$newdir" ]] && err_exit "directory $newdir allready exists, I refuse to run!" 1

shift

providetext="Provides:\tultragrid-proprietary-drivers-release-$lockver"
conflicttext="Conflicts:"
sep="\t"
for version in $(printf "%s" "$*" | sed -r 's#\b([[:digit:].]+)#release-\1#g') nightly ; do
	conflicttext+="${sep}ultragrid-proprietary-drivers-$version"
	sep=", "
done

cp -r "$scriptroot/$driverssubdir" "$newdir"

pushd "$newdir"

debdirpatch "debian." "" "$lockver" "$conflicttext" "$providetext"

for debarchive in debian*.tar.gz ; do
	debdirpatch "debian/" "$debarchive" "$lockver"  "$conflicttext" "$providetext"
done

#patch package name in rpmspec
for specfile in *.spec ; do
	test -f "$specfile" || continue
	sed -i "s#-rpmlintrc\$#-$lockver-rpmlintrc#g" "$specfile"
	patch_file "$conflicttext\n$providetext" "$specfile"
	patch_pkgname "$specfile" "$lockver"
done
#patch package name in debian file
for dscfile in *.dsc ; do
	test -f "$dscfile" || continue
	patch_pkgname "$dscfile" "$lockver"
done

# process rpmlintrc
for lintfile in *-rpmlintrc ; do
	test -f "$lintfile" || continue
	sed -i "s#ultragrid-proprietary-drivers#ultragrid-proprietary-drivers-$lockver#g" "$lintfile"
done

for lintfile in *rpmlintrc *.spec *.dsc ; do
	test -f "$lintfile" || continue
	move_file "$lintfile" "$lockver"
done

for servicefile in _service* ; do
	sed -i "s#package_specs/ultragrid-proprietary-drivers/#package_specs/ultragrid-proprietary-drivers-$lockver/#g" "$servicefile"
done
popd
