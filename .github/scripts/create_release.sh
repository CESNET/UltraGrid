#!/bin/sh -eux

## == NEWS/FIXES.md processing description (release)
## NEWS: 1. first line is stripped; then all content is used until new line
## FIXES.md: if present, whole file is used
## both:
## a. if there is a line starting with space, it is appended to previous one (markdown)
## b. <NL> are replaced with '\n' (to create valid JSON)
## c. introductionary text ("Changes:" etc) is added to the string
## d. backlash in MD special escape sequences ('\_', \*') are escaped (again to gain valid JSON)

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/json-common.sh"

# Joins line that starts with space to previous:
# https://www.gnu.org/software/sed/manual/html_node/Joining-lines.html
MERG_LN=':a ; $!N ; s/\n\s+/ / ; ta ; P ; D'
# https://unix.stackexchange.com/questions/114943/can-sed-replace-new-line-characters
REPL_NL=':a;N;$!ba;s/\n/\\n/g'
markdown_replaces='s/\\_/\\\\_/g;s/\\\*/\\\\*/g' # escape MD escape sequences

sudo apt install jq
URL=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET\
 "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG" |
  jq -r '.url')
REQ=PATCH
if [ "$URL" = null ]; then # release doesn't yet exist
  REQ=POST
  URL=https://api.github.com/repos/$GITHUB_REPOSITORY/releases
fi
DATE=$(date -Iminutes)
if [ "$VERSION" = continuous ]; then
        TITLE='continuous builds'
        SUMMARY="Current builds from Git master branch. macOS alternative\
 build is rebuilt daily, Linux ARM builds weekly. Archived builds can be\
 found [here](https://frakira.fi.muni.cz/~xpulec/ug-nightly-archive/),\
 additional alternative builds\
 [here](https://frakira.fi.muni.cz/~xpulec/ug-alt-builds/)."
        PRERELEASE=true
else
        TITLE="UltraGrid $VERSION"
        FIXES=
        if [ -f FIXES.md ]; then
                TMP=$(mktemp)
                sed -E "$MERG_LN" < FIXES.md | sed -e "$REPL_NL" > "$TMP"
                F=$(cat "$TMP")
                FIXES="### Fixes since last release:\n$F\n"
                rm "$TMP"
        fi
        TMP=$(mktemp)
        sed -e '1,2d' -e '/^$/q' < NEWS | sed -E "$MERG_LN" | sed -e "$REPL_NL" > "$TMP"
        N=$(cat "$TMP")
        SUMMARY=$(printf "%s" "### Changes:\n$N\n$FIXES\n**Full changelog:** https://github.com/$GITHUB_REPOSITORY/commits/$TAG" | sed -e "$markdown_replaces")
        rm "$TMP"
        PRERELEASE=false
fi

tmp=$(mktemp)
status=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X "$REQ" "$URL" -T - -o "$tmp" -w '%{http_code}' <<EOF
{
  "tag_name": "$TAG", "name": "$TITLE",
  "body": "Built $DATE\n\n$SUMMARY",
  "draft": false, "prerelease": $PRERELEASE
}
EOF
)

check_status "$status" "$tmp"
rm "$tmp"

