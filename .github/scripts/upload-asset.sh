#!/bin/sh -eu

TAG_NAME=${1?}
FILE=${2?}
CONTENT_TYPE=${3?}
LABEL=${4?}

UPLOAD_URL=$(curl -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/nightly | jq -r .upload_url | sed "s/{.*}//")

curl -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $CONTENT_TYPE" -X POST "$UPLOAD_URL?name=$FILE&label=$LABEL" -T $FILE
