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

check_type() {
        TYPE=$(echo "$1" | jq -r type)
        if [ "$TYPE" != "$2" ]; then
                echo "Wrong JSON type - expected $2, got $TYPE" >&2
                echo "JSON: $JSON" >&2
                exit 1
        fi
}

## @brief Returns json for given URL and authorization token while checking errors
## @param $1 URL
## @param $2 GITHUB_TOKEN
## @param $3 requested type (optional)
fetch_json() {
        TMPNAM=$(mktemp)
        STATUS=$(curl -S -H "Authorization: token ${2?GitHub token is required}" -X GET ${1?URL is required} -w "%{http_code}" -o $TMPNAM)
        JSON=$(cat $TMPNAM)
        rm $TMPNAM
        if [ $STATUS -ne 200 ]; then
                echo "HTTP error code $STATUS" >&2
                echo "JSON: $JSON" >&2
        fi
        check_errors "$JSON"
        if [ -n ${3-""} ]; then
                check_type "$JSON" $3
        fi
        echo "$JSON"
}

## @brief Checks HTTP error code for success
## @param $1 returned HTTP status code
## @param $2 (optional) returned JSON (to be printed in case of error)
check_status() {
        if [ $1 -lt 200 -o $1 -ge 300 ]; then
                echo "Wrong response status $STATUS!" >&2
                if [ -n ${2-""} ]; then
                        echo "JSON: $JSON" >&2
                fi
                exit 1
        fi
}

