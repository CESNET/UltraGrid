#!/bin/sh -eux
# Prints to stdout ETags for given URLs as parameters in format ETag1_Etag2[_Etagn].
# If failed to optain, the particular tag is replaced with a keyword NOTFOUND

OUT=

while [ $# -gt 0 ]; do
        ETAG=$(curl -LI $1 | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/')
        OUT=$OUT${OUT:+_}${ETAG:-NOTFOUND}
        shift
done

echo $OUT

