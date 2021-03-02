#!/bin/sh

SILENT=

while getopts 'hs' opt; do
	case "$opt" in
		'h'|'?')
			cat <<-EOF
			Usage:
			    $0 [-s] [STACKTRACE_FILE]
			where
			          -s        - silent
			    STACKTRACE_FILE - file containg stacktrace from UG (if not set, stdin is used)
			EOF
			[ $opt = h ] && exit 0 || exit 1
			;;
		's')
			SILENT=yes
			;;
	esac
done

shift $(($OPTIND - 1))

IN=${1-/dev/stdin}

while read n; do
	if ! expr "$n" : ".*(.*\+.*)" > /dev/null; then
		continue
	fi
	EXE=$(expr $n : "\([^(]*\)")
	addr=$(expr $n : ".*(\(.*\))")
	# addr in format "func+addr"
	if expr $addr : "[^\+]" >/dev/null; then
		FUNC=$(expr $addr : "\([^\+]*\)")
		OFFSET=$(expr $addr : "[^\+]*\+\(.*\)")
		LINE=$(objdump -d -F $EXE | grep "<$FUNC>" | grep -v '^ ')
		FUNC_OFFSET=$(expr "$LINE" : '.*File Offset: \([^)]*\)')
		addr=$(($FUNC_OFFSET + $OFFSET))
		addr=$(printf 0x%x $addr) # addr2line requires hex
	fi
	[ -z $SILENT ] && printf "Decoding $n\n"
	ARGS=
	[ -z $SILENT ] && ARGS=-f
	addr2line -e $EXE $ARGS -C $addr
done <$IN

# vim: set noexpandtab:
