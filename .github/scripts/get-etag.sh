#!/bin/sh -ux
#
## Prints GH variable $1 equal to ETag of URL in $2. The output is supposed
## to be redirected to $GITHUB_OUTPUT file.
##
## If failed, it is sets $1 to empty string - the reason is to allow
## partial matching of GH actions/cache.

output=$1
url=$2

ETAG=$(curl -ILf "$url" | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/')
printf '%s=' "$output"
if [ "$ETAG" ]; then
        printf '%s\n' "$ETAG" | sed 's/[^-._A-Za-z0-9]/_/g'
else
        printf '\n'
fi
