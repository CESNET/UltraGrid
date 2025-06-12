#!/bin/sh -eu
#
## downloads gh release asset
## @param $1 GH repo (eg aja-video/ntv2)
## @param $2 file pattern to download
## @param $3 output file name

dir=$(dirname "$0")
# shellcheck source=/dev/null
. "$dir/json-common.sh"

readonly repo="${1?repo must be given}"
readonly pattern="${2?pattern must be given!}"
readonly file="${3?output filename must be given!}"

gh_release_json=$(fetch_json "https://api.github.com/repos/$repo/releases" \
        "${GITHUB_TOKEN-}" array)
gh_path=$(jq  -e -r '
[ .[]                                    # for each top‐level object
  | .assets[]                            # walk into its assets array
  | select(.name | test("'"$pattern"'")) # keep only those whose name matches
  | .browser_download_url                # print just the URL
][0]                                     # include only 1st match
' "$gh_release_json")
rm -- "$gh_release_json"
if [ -n "${GITHUB_TOKEN-}" ]; then
        set -- -H "Authorization: token $GITHUB_TOKEN"
else
        set --
fi
curl -sSL "$@" "$gh_path" -o "$file"

