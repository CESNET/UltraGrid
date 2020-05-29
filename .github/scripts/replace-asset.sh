#!/bin/sh -eux

set +e # pattern matching may fail
# If first parameter 2 parameters is GITHUB_REPOSOTIRY and GITHUB_TOKEN, those willbe used as an env var (used by the scripts below)
REPOSITORY=$(expr $1 : "GITHUB_REPOSITORY=\(.*\)")
if [ $? -eq 0 ]; then
        export GITHUB_REPOSITORY=$REPOSITORY
        shift
fi
TOKEN=$(expr $1 : "GITHUB_TOKEN=\(.*\)")
if [ $? -eq 0 ]; then
        export GITHUB_TOKEN=$TOKEN
        shift
fi
set -e

DIR=$(dirname $0)

$DIR/delete-asset.sh "$@"
$DIR/upload-asset.sh "$@"

