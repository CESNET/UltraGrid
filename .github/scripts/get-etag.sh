#!/bin/sh -ux
# Prints to stdout ETag for given URL.
# If failed to obtain and $2=optional the particular tag is replaced with a keyword NOTFOUND

ETAG=$(curl -LI "$1" | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/')
if [ "$ETAG" ]; then
        printf '%s\n' "$ETAG"
        exit 0
fi

if [ "${2-}" = optional ]; then
        echo NOTFOUND
        exit 0
fi

exit 1
