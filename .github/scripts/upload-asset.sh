#!/bin/sh -eu

. $(dirname $0)/json-common.sh

TAG_NAME=${1?}
FILE=${2?}
CONTENT_TYPE=${3?}
LABEL=${4?}

JSON=$(fetch_json https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME $GITHUB_TOKEN)
UPLOAD_URL=$(echo "$JSON" | jq -r .upload_url | sed "s/{.*}//")

TMPNAME=$(mktemp)
STATUS=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $CONTENT_TYPE" -X POST "$UPLOAD_URL?name=$FILE&label=$LABEL" -T $FILE -w %{http_code} -o $TMPNAME)
JSON=$(cat $TMPNAME)
rm $TMPNAME
check_errors "$JSON"
check_status $STATUS

