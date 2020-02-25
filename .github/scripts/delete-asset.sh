#!/bin/sh -eux

check_errors() {
        TYPE=$(echo "$1" | jq -r type)
        if [ "$TYPE" != object ]; then
                return
        fi
        ERRORS=$(echo "$1" | jq -r '.errors')
        if [ "$ERRORS" != null ]; then
                echo $ERRORS >&2
                exit 1
        fi
}

check_type() {
        TYPE=$(echo "$1" | jq -r type)
        if [ "$TYPE" != "$2" ]; then
                return
        fi
}

TAG_NAME=${1?}
PATTERN=${2?}
JSON=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME)
check_errors "$JSON"
RELEASE_ID=$(echo "$JSON" | jq -r '.id')
JSON=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/$RELEASE_ID/assets)
check_errors "$JSON"
check_type "$JSON" array 
LEN=$(echo "$JSON" | jq length)
for n in `seq 0 $(($LEN-1))`; do
        NAME=$(echo "$JSON" | jq '.['$n'].name')
        if expr "$NAME" : "\"$PATTERN\"$"; then
                ID=$(echo "$JSON" | jq '.['$n'].id')
                JSON=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X DELETE "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/assets/$ID")
                check_errors "$JSON"
        fi
done

