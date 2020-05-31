#!/bin/sh -eux

. $(dirname $0)/json-common.sh

TAG_NAME=${1?}
PATTERN=$(basename "${2?}")
JSON=$(fetch_json https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME $GITHUB_TOKEN)
RELEASE_ID=$(jq -r '.id' $JSON)
rm $JSON
JSON=$(fetch_json https://api.github.com/repos/$GITHUB_REPOSITORY/releases/$RELEASE_ID/assets $GITHUB_TOKEN array)
LEN=$(jq length $JSON)
for n in `seq 0 $(($LEN-1))`; do
        NAME=$(jq -r '.['$n'].name' "$JSON")
        if expr "$NAME" : "$PATTERN$"; then
                ID=$(jq '.['$n'].id' $JSON)
                JSON2=$(mktemp)
                STATUS=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X DELETE "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/assets/$ID" -w %{http_code} -o $JSON2)
                check_errors "$JSON2"
                check_status $STATUS
                rm $JSON2
        fi
done
rm $JSON

