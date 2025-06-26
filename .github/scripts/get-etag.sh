#!/bin/sh -ux
# Prints to stdout ETag for given URL.
# If failed to obtain and $2=optional the particular tag is replaced with a keyword NOTFOUND

output=$1
url=$2
optional=${3-}

ETAG=$(curl -LI "$url" | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/')
if [ "$ETAG" ]; then
        printf '%s=' "$output"
        printf '%s\n' "$ETAG" | sed 's/[^-._A-Za-z0-9]/_/g'
        exit 0
fi

if [ "$optional" = optional ]; then
        echo "$1=NOTFOUND"
        exit 0
fi

exit 1
