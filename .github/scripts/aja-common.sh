# shellcheck shell=sh
download_aja_release_asset() {
        pattern=${1?pattern must be given!}
        file=${2?output filename must be given!}
        aja_release_json=$(fetch_json https://api.github.com/repos/aja-video/ntv2/releases "${GITHUB_TOKEN-}" array)
        aja_gh_release=$(jq -r '.[0].assets_url' "$aja_release_json")
        rm -- "$aja_release_json"
        aja_gh_path_json=$(fetch_json "$aja_gh_release" "${GITHUB_TOKEN-}" array)
        aja_gh_path=$(jq -r '[.[] | select(.name | test(".*'"$pattern"'.*"))] | .[0].browser_download_url' "$aja_gh_path_json")
        rm -- "$aja_gh_path_json"
        curl -sSL "$aja_gh_path" -o "$file"
}

