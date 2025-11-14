#!/bin/sh -eux
#
# Ensures that tag "continuous" is present is released on GH. This is required for
# zsync AppImage release asset.
#

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/json-common.sh"

sudo apt install jq
URL=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET\
 "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/continuous" |
 jq -r '.url')
if [ "$URL" != null ]; then # release exists
        exit 0
fi
git fetch --prune --unshallow --tags -f
if git tag continuous; then # tag may or may not exists
        git push origin refs/tags/continuous
fi
tmp=$(mktemp)
status=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X POST "$URL" -T - -o "$tmp" -w '%{http_code}' <<EOF
{
  "tag_name": "continuous"}
}
EOF
)

check_status "$status" "$tmp"
rm "$tmp"

