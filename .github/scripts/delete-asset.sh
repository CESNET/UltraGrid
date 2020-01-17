#!/bin/sh

TAG_NAME=${1?}
PATTERN=${2?}
RELEASE_ID=$(curl -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME | jq -r '.id')
curl -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/$RELEASE_ID/assets > assets.json
LEN=`jq "length" assets.json`
for n in `seq 0 $(($LEN-1))`; do
        NAME=`jq '.['$n'].name' assets.json`
        if expr "$NAME" : "\"$PATTERN\"$"; then
                ID=`jq '.['$n'].id' assets.json`
                curl -H "Authorization: token $GITHUB_TOKEN" -X DELETE "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/assets/$ID"
        fi
done

