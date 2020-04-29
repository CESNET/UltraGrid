#!/bin/sh -eux

. $(dirname $0)/json-common.sh

TAG_NAME=${1?}
PATTERN=${2?}
JSON=$(fetch_json https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME $GITHUB_TOKEN)
RELEASE_ID=$(echo "$JSON" | jq -r '.id')
JSON=$(fetch_json https://api.github.com/repos/$GITHUB_REPOSITORY/releases/$RELEASE_ID/assets $GITHUB_TOKEN array)
LEN=$(echo "$JSON" | jq length)
for n in `seq 0 $(($LEN-1))`; do
        NAME=$(echo "$JSON" | jq -r '.['$n'].name')
        if expr "$NAME" : "$PATTERN$"; then
                ID=$(echo "$JSON" | jq '.['$n'].id')
                TMPNAME=$(mktemp)
                STATUS=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X DELETE "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/assets/$ID" -w %{http_code} -o $TMPNAME)
                JSON=$(cat $TMPNAME)
                rm $TMPNAME
                check_errors "$JSON"
                check_status $STATUS
        fi
done

