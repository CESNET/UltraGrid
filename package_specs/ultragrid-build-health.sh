#!/bin/sh
#Script for nightly build

function is_build_enabled() {
	threshold=$((14*24*60*60))
	repo="$1"
	for arch in $(osc repos | grep -E "\b$repo\b" | sed -r -e 's#^[^\s]+ ##g') ; do
		builddate=$(osc buildhist  --limit 3 "$repo" "$arch" | tail -n +2 | tail -n 1 | sed -r 's#^([0-9-]+\s[0-9:-]+).*$#\1#g')
		if test -z "$builddate" ; then
			continue
		fi

		builddate=$(date -d "$builddate" '+%s')
		if test $(($builddate + $threshold)) -gt $(date '+%s') ; then
			return 0
		fi
	done

	return 1
}

function makelof(){
	repo="$1"
	rm binaries/$repo/.fulllist
	for file in binaries/$repo/* ; do
		if ! test -f $file ; then
			continue
		fi

		if [[ $file =~ .*\.deb$ ]] ; then
			dpkg -c "$file" | sed -r 's#^([^\s]+\s){5}##g' | grep -vE '(\.debug$)|(-dbgsym\b)' >> binaries/$repo/.fulllist
		elif [[ $file =~ .*\.rpm$ ]] ; then
			rpm -ql -p "$file" | grep -vE '/\.build-id/' >> binaries/$repo/.fulllist
		fi
	done

	if ! test -f binaries/$repo/.fulllist ; then
		return 1
	fi

	cat binaries/$repo/.fulllist | sed -r -e 's#^((\.?/)?([^/]+/)+)[^/]+$#\1#g' -e 's#/$##g' | uniq > binaries/$repo/.dirlist
	cat binaries/$repo/.fulllist | sed -r -e 's#/$##g' | fgrep -vxf binaries/$repo/.dirlist > binaries/$repo/.files

	rm binaries/$repo/.dirlist binaries/$repo/.fulllist
	sed -r 's#^.*/([^/]+)$#\1#g' binaries/$repo/.files | sort
	rm binaries/$repo/.files
	return 0
}

function makereference(){
	for file in pkghealth/* ; do 
		printf "%d %s\n" $(cat "$file" | wc -l) "$file" >> pkghealth/_sort
	done
	sort -rn pkghealth/_sort | head -n 1 | sed -r 's#^[0-9]+\s##g'
}

function gencomp(){
	reference="$1"
	for stat in pkghealth/* ; do
		reponame=$(echo $stat | cut -d/ -f2)
		comm -2 -3 $stat $reference > stats/$reponame.extras
		comm -1 -3 $stat $reference > stats/$reponame.missing
	done
}

function launch() {
	PACKAGE="$1"

	pushd "$PACKAGE"
	osc update
	#osc service localrun

	rm -rf pkghealth
	mkdir -p pkghealth

	for repo in $(osc repos | cut -d' ' -f1); do 
		if ! is_build_enabled "$repo" ; then
			continue;
		fi
	
		rm -rf binaries/$repo
		mkdir -p binaries/$repo
		osc getbinaries -q -d binaries/$repo $repo
		makelof $repo  > pkghealth/$repo
	done

	reference=$(makereference)
	rm pkghealth/_sort*

	rm -rf stats
	mkdir -p stats

	gencomp $reference

	for statfile in $(ls -1 stats/* | sort ) ; do
		if ! test -s "$statfile" ; then
			continue
		fi

		printf "%s:\n" "$statfile"
		cat $statfile | while read line ; do
			printf "\t%s\n" $line
		done
	done

	rm -rf pkghealth binaries stats

	return 0
}

function usage() {
	printf "\t-b BASE\t\tset base directory, which all other directories will be addressed relative to; defaults to \$HOME\n"
	printf "\t-p PACKAGE\t\tset package path, either as absolute or relative to \$BASE; defaults to \$(basename \$PROJECT)\n"
	printf "\t-h print this help\n"
}

function launch_diff() {
	PACKAGE="$1"
	DIFFFILE="$2"
	if [ -n "$DIFFFILE" ] ; then
		tmpfile=$(mktemp)
		exval=0
		mkdir -p "$(dirname "$DIFFFILE")"
		touch "$DIFFFILE"
		launch "$PACKAGE" >"$tmpfile" 2>&1 || exval=$?
		diff "$tmpfile" "$DIFFFILE"
		xv=$?
		exval=$(($exval + $xv))
		mv "$tmpfile" "$DIFFFILE"
		return $exval
	else
		launch "$PACKAGE"
	fi
}

# defaults
BASE=""
PACKAGE=""
DIFFFILE=""

ARGS=$(getopt -o "b:p:d:h" -- "$@")
eval set -- "$ARGS"

while [ "x$1" != "x--" ] ; do
	case "$1" in
	-b)	BASE="$2" ;;
	-p)	PACKAGE="$2" ;;
	--diff | -d)	DIFFFILE="$2" ;;
	-h | *)	usage ; exit 1 ;;
	esac
	shift 2
done

#echo $BASE $PACKAGE $REF $PROJECT $SPECDIR

# complete with defaults
[[ -z "$BASE" ]] && BASE="$HOME"
[[ -z "$PACKAGE" ]] && PROJECT="ultragrid-nightly"
#[[ -z "$DIFFILE" ]]

absolute_path_test_regex="^/.*$"

# absolutize
[[ "$BASE" =~ $absolute_path_test_regex ]] || BASE="$(realpath "${BASE}")"
[[ "$PACKAGE" =~ $absolute_path_test_regex ]] || PACKAGE="${BASE}/${PACKAGE}"

if tty -s ; then
	launch_diff "$PACKAGE" "$DIFFFILE" 
else
	tmpfile=$(mktemp)
	launch_diff "$PACKAGE" "$DIFFFILE" >"$tmpfile" 2>&1 || cat "$tmpfile"
	rm "$tmpfile"
fi

