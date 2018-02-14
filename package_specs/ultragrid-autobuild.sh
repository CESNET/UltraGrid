#!/bin/sh
#Script for nightly build

function launch() {
	PACKAGE="$1"
	REF="$2"

	PROJECT="$3"
	SPECDIR="$4"

	pushd "$PROJECT"
	OLDHEAD=$(git rev-parse $REF )

	# tags have to be erased first
	if (git tag | grep "^${REF}\$" > /dev/null 2>&1) ; then
		git tag -d "$REF"
	fi

	git fetch origin

	# branches need remote prepended
	if (git branch | grep -E "^[[:space:]*]*${REF}\$" > /dev/null 2>&1) ; then
		git checkout "$REF"
		git reset --hard origin/"$REF"
	else
		git reset --hard "$REF"
	fi

	NEWHEAD=$(git rev-parse $REF )

	echo $OLDHEAD $NEWHEAD

	#check if $PROJECT is up-to-date
	if [ "$OLDHEAD" = "$NEWHEAD" ]; then
		return 1
	fi

	DATE=$(git log --date=short "$REF" | head -n 3 | tail -n 1 | sed 's/[Date: -]//g')

	popd
	pushd "$PACKAGE"
	osc update
	#osc service localrun

	for spec in $(echo ${SPECDIR}/*.dsc.tpl) ; do
		sed "/Version/s/-.*/-${DATE}00/" "$spec" > "${PACKAGE}/$(basename "$spec" | sed -r 's/^([^:]+[:])*([^:]+).tpl$/\2/g')"
	done

	for spec in $(echo ${SPECDIR}/*.spec.tpl) ; do
		sed -r "/Release:/s/([[:space:]]+)[[:digit:].]+$/\\1${DATE}.00/" "$spec" > "${PACKAGE}/$(basename "$spec" | sed -r 's/^([^:]+[:])*([^:]+).tpl$/\2/g')"
	done

	sed -r --in-place "s#^.* dummy commit .*\$#<!-- dummy commit $NEWHEAD -->#g" _service

	for file in *.spec *.dsc ; do
		osc add $file
	done

	osc commit -m "Automated commit - $NEWHEAD"

	popd
	echo "Script OK - $NEWHEAD"

	return 0
}

function usage() {
	printf "\t-b BASE\t\tset base directory, which all other directories will be addressed relative to; defaults to \$HOME\n"
	printf "\t-p PACKAGE\t\tset package path, either as absolute or relative to \$BASE; defaults to \$(basename \$PROJECT)\n"
	printf "\t-r REF\t\tset branch / tag / commit to build package; defaults to nightly\n"
	printf "\t-P PROJECT\t\tset path to git repository clone, either as absolute or relative to \$BASE; defaults to ultragrid-nightly\n"
	printf "\t-s SPECDIR\t\tset path to package specification templates, either as absolute or relative to \$PROJECT; defaults to package_specs/\$(basename \$PACKAGE)\n"
	printf "\t-h print this help\n"
}

# defaults
BASE=""

PACKAGE=""
REF=""
PROJECT=""
SPECDIR=""

ARGS=$(getopt -o "b:p:r:P:s:h" -- "$@")
eval set -- "$ARGS"

while [ "x$1" != "x--" ] ; do
	case "$1" in
	-b)	BASE="$2" ;;
	-p)	PACKAGE="$2" ;;
	-r)	REF="$2" ;;
	-P)	PROJECT="$2" ;;
	-s)	SPECDIR="$2" ;;
	-h | *)	usage ; exit 1 ;;
	esac
	shift 2
done

#echo $BASE $PACKAGE $REF $PROJECT $SPECDIR

# complete with defaults
[[ -z "$BASE" ]] && BASE="$HOME"
[[ -z "$REF" ]] &&  REF="nightly"
[[ -z "$PROJECT" ]] && PROJECT="ultragrid-nightly"
[[ -z "$PACKAGE" ]] && PACKAGE="sitola/$(basename "${PROJECT}")"
[[ -z "$SPECDIR" ]] && SPECDIR="package_specs/$(basename "$PACKAGE")"

absolute_path_test_regex="^/.*$"

# absolutize
[[ "$BASE" =~ $absolute_path_test_regex ]] || BASE="$(realpath "${BASE}/${BASE}")"
[[ "$PACKAGE" =~ $absolute_path_test_regex ]] || PACKAGE="${BASE}/${PACKAGE}"
[[ "$PROJECT" =~ $absolute_path_test_regex ]] || PROJECT="${BASE}/${PROJECT}"
[[ "$SPECDIR" =~ $absolute_path_test_regex ]] || SPECDIR="${PROJECT}/${SPECDIR}"

if tty -s ; then
	launch "$PACKAGE" "$REF" "$PROJECT" "$SPECDIR"
else
	tmpfile=$(mktemp)
	launch "$PACKAGE" "$REF" "$PROJECT" "$SPECDIR" >"$tmpfile" 2>&1 || cat "$tmpfile"
	rm "$tmpfile"
fi

