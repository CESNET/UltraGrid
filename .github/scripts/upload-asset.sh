#!/bin/sh -eu

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

TAG_NAME=${1?}
FILE=${2?}
CONTENT_TYPE=${3?}
LABEL=${4?}

JSON=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -X GET https://api.github.com/repos/$GITHUB_REPOSITORY/releases/tags/$TAG_NAME)
check_errors "$JSON"
UPLOAD_URL=$(echo "$JSON" | jq -r .upload_url | sed "s/{.*}//")

JSON=$(curl -S -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $CONTENT_TYPE" -X POST "$UPLOAD_URL?name=$FILE&label=$LABEL" -T $FILE)
check_errors "$JSON"

