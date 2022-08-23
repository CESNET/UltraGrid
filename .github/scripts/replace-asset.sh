#!/bin/sh -eux

set +e # pattern matching may fail
# If first parameter 2 parameters is GITHUB_REPOSOTIRY and GITHUB_TOKEN, those willbe used as an env var (used by the scripts below)
if repository=$(expr "$1" : "GITHUB_REPOSITORY=\(.*\)"); then
        export GITHUB_REPOSITORY="$repository"
        shift
fi
if token=$(expr "$1" : "GITHUB_TOKEN=\(.*\)"); then
        export GITHUB_TOKEN="$token"
        shift
fi
set -e

dir=$(dirname "$0")

"$dir/delete-asset.sh" "$@"
"$dir/upload-asset.sh" "$@"

