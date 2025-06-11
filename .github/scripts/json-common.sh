# shellcheck shell=sh
is_int() { test "$@" -eq "$@"; }

check_errors() {

        TYPE=$(jq -r type "$1")
        if [ "$TYPE" != object ]; then
                return
        fi
        ERRORS=$(jq -r '.errors' "$1")
        if [ "$ERRORS" != null ]; then
                echo "$ERRORS" >&2
                exit 1
        fi
}

check_type() {
        TYPE=$(jq -r type "$1")
        if [ "$TYPE" != "$2" ]; then
                echo "Wrong JSON type - expected $2, got $TYPE" >&2
                echo 'JSON:' >&2
                cat "$1" >&2
                exit 1
        fi
}

## @brief Returns json file for given URL and authorization token while checking errors
## @param $1 URL
## @param $2 GITHUB_TOKEN (optional)
## @param $3 requested type (optional)
fetch_json() {
        json=$(mktemp)
        url=${1?URL is required}
        github_token=${2-}
        req_type=${3-}
        if [ -n "$github_token" ]; then
                set -- -H "Authorization: token $github_token"
        else
                set --
        fi
        status=$(curl -sS "$@" -X GET "$url" -w "%{http_code}" -o "$json")
        if ! is_int "$status" || [ "$status" -ne 200 ]; then
                echo "HTTP error code $status" >&2
                echo "JSON:" >&2
                cat "$json" >&2
        fi
        check_errors "$json"
        if [ -n "$req_type" ]; then
                check_type "$json" "$req_type"
        fi
        echo "$json"
}

## @brief Checks HTTP error code for success
## @param $1 returned HTTP status code
## @param $2 (optional) returned JSON (to be printed in case of error)
check_status() {
        if ! is_int "$1" || [ "$1" -lt 200 ] || [ "$1" -ge 300 ]; then
                echo "Wrong response status $1!" >&2
                if [ -n "${2-}" ]; then
                        echo "JSON:" >&2
                        cat "$2"
                fi
                exit 1
        fi
}

