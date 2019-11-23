#!/bin/bash
#
# Generates (prints) list of DLL dependencies of Windows executable
# (would work with DLL itself as well).
#
# TODO: check if there isn't any problem with paths with spaces
# TODO: exclude only /c/Windows?

# prints dependecies
get_depends() {
        if [ -f "$1" ]; then
                objdump -x "$1" | grep 'DLL Name' | awk '{ print $3 }'
        fi
}

# locates appropriate DLL in PATH
find_dll() {
        IFS=:
        for n in $PATH; do
                if [ -f "$n/$1" ]; then
                        echo $n/$1
			break
                fi
        done
}

is_not_system_dll() {
        return $(test $(expr match "$1" '^/c/[Ww]indows') -eq 0)
}

if [ $# -eq 0 ]; then
	echo "Usage $0 <executable>"
	echo -e "\tRecursively generates dependencies of the executable (DLLs)."
	echo -e "\tExcludes paths containing /c/Windows (system DLLs) in MSYS path syntax."
	exit 1
fi

declare -A DONE # list of already processed DLLs

INITIAL=$1 # save root element to skip the program itself

while test $# -gt 0; do
        # skip already processed item
        if [ -n "${DONE[$1]}" ]; then
                shift
                continue
        fi
        # push dependencies of current item to stack
        for n in `get_depends "$1"`; do
                DLL=`find_dll "$n"`
                if [ -z "$DLL" ]; then
                        continue
                fi
                if test -z "${DONE[$DLL]}" && is_not_system_dll "$DLL"; then
                        echo "Adding $DLL" >&2
                        set -- "$@" "$DLL"
                else
                        echo "Not adding $DLL" >&2
                fi
        done
        # print the item (omit initial)
        if [ "$1" != "$INITIAL" ]; then
                echo $1
        fi
        DONE[$1]=1
        shift
done

