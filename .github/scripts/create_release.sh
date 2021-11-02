#!/bin/sh -eu

sudo apt install jq
URL=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG | jq -r '.url')
REQ=PATCH
if [ $URL = null ]; then # release doesn't yet exist
  REQ=POST
  URL=https://api.github.com/repos/$GITHUB_REPOSITORY/releases
fi
DATE=$(date -Iminutes)
if [ $VERSION = continuous ]; then
        TITLE='continuous builds'
        SUMMARY='Current builds from Git master branch. macOS alternative build is rebuilt daily, Linux ARM builds occasionally. Archived builds can be found [here](https://frakira.fi.muni.cz/~xpulec/ug-nightly-archive/).'
        PRERELEASE=true
else
        TITLE="UltraGrid $VERSION"
        SUMMARY="**Short log:**$(cat NEWS | sed -e '1,2d' -e '/^$/q' -e 's/^\*/\\n*/' | tr '\n' ' ')\n\n**Full changelog:** https://github.com/MartinPulec/UltraGrid/commits/$TAG"
        PRERELEASE=false
fi

curl -S -H "Authorization: token $GITHUB_TOKEN" -X $REQ $URL -T - <<EOF
{
  "tag_name": "$TAG", "name": "$TITLE",
  "body": "Built $DATE\n\n$SUMMARY",
  "draft": false, "prerelease": $PRERELEASE
}
