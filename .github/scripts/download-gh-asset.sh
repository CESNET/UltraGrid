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
gh_release=$(jq -r '.[0].assets_url' "$gh_release_json")
rm -- "$gh_release_json"
gh_path_json=$(fetch_json "$gh_release" "${GITHUB_TOKEN-}" array)
gh_path=$(jq -r '[.[] | select(.name | test(".*'"$pattern"'.*"))] |
        .[0].browser_download_url' "$gh_path_json")
rm -- "$gh_path_json"
curl -sSL "$gh_path" -o "$file"

