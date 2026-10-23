#!/bin/sh -ux
#
## Prints GH variable $1 equal to ETag of URL in $2. The output is supposed
## to be redirected to $GITHUB_OUTPUT file.
##
## If ETag is not provided (some servers no longer provide), use content-length
## as a workaround.
##
## If failed, it is sets $1 to empty string - the reason is to allow
## partial matching of GH actions/cache.

output=$1
url=$2

printf '%s=' "$output"

# if server is down, just return an empty string
if ! resp=$(curl -IL "$url"); then
        printf '\n'
        exit 0
fi

file_info=$(printf '%s' "$resp" | tr -d '\r')

etag=$(printf '%s' "$file_info" | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/')
if [ "$etag" ]; then
        printf '%s\n' "$etag" | sed 's/[^-._A-Za-z0-9]/_/g'
        exit 0
fi
content_length=$(printf '%s' "$file_info" | grep -i '^content-length' |
        awk '{print $2}')
if [ "$content_length" ]; then
        printf "content-length-%d\n" "$content_length"
        exit 0
fi

printf "Neither ETag nor Content-Length provided, got:\n%s\n" "$file_info"
exit 1
