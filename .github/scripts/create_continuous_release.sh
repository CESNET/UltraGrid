#!/bin/sh -eux
#
# Ensures that tag "continuous" is present is released on GH. This is required for
# zsync AppImage release asset.
#

sudo apt install jq
URL=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG" | jq -r '.url')
if [ "$URL" != null ]; then # release exists
        exit 0
fi
git fetch --prune --unshallow --tags -f
git tag continuous && git push origin refs/tags/continuous
curl -S -H "Authorization: token $GITHUB_TOKEN" -X POST "$URL" -T - <<EOF
{
  "tag_name": "continuous"}
}
EOF
