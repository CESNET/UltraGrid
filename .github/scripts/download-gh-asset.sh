#!/bin/sh -eu
#
## downloads gh release asset
## @param $1 GH repo (eg aja-video/ntv2)
## @param $2 file pattern to download
## @param $3 output file name (optional)

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/json-common.sh"

readonly repo="${1?repo must be given}"
readonly pattern="${2?pattern must be given!}"
readonly file="${3-}"
readonly json_path="https://api.github.com/repos/$repo/releases"

gh_release_json=$(fetch_json "$json_path" "${GITHUB_TOKEN-}" array)
set +e
gh_path=$(jq  -e -r '
[ .[]                                    # for each top‐level object
  | .assets[]                            # walk into its assets array
  | select(.name | test("'"$pattern"'")) # keep only those whose name matches
  | .browser_download_url                # print just the URL
][0]                                     # include only 1st match
' "$gh_release_json")
rm -- "$gh_release_json"
if [ "$gh_path" = null ]; then
        echo "Pattern ${pattern?} not found in JSON from ${json_path?}"
        exit 1
fi
set -e

if [ -n "${GITHUB_TOKEN-}" ]; then
        set -- -H "Authorization: token $GITHUB_TOKEN"
else
        set --
fi
if [ "${file-}" ]; then
        echo "Downloading $gh_path to $file" 1>&2
        set -- "$@" -o "$file"
else
        echo "Downloading $gh_path" 1>&2
        set -- "$@" -O
fi
curl -LSfs "$@" "$gh_path"
