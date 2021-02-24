## fetches a submodule
##
## @param $1 name
## @param $2 fallback Gzip URL
## @param $3 fallback Git URL
## @retval 1 submodule was updated
fetch_submodule() {
        MODULE=$1
        FALLBACK_URL=$2
        FALLBACK_GIT_URL=$3
        SUBMODULE_UPDATED= # assume that submodule have not been updated
        printf "Downloading ${MODULE}... "
        if ! command -v git >/dev/null; then
                if [ ! -d ext-deps/$MODULE ]; then
                        echo "git not found - trying to download a release"
                        mkdir -p ext-deps/tmp
                        curl -L $FALLBACK_URL | tar -C ext-deps/tmp -xz
                        mv ext-deps/tmp/* ext-deps/$MODULE
                        rmdir ext-deps/tmp
                else
                        echo "skipped (ext-deps/$MODULE is already present)"
                fi
        elif ! git branch >/dev/null 2>&1; then
                # we are not in UltraGrid git repository but can use git to download
                if [ ! -d ext-deps/$MODULE ]; then
                        echo "git submodule found - trying to clone"
                        git clone $FALLBACK_GIT_URL ext-deps/$MODULE
                else
                        echo "skipped (ext-deps/$MODULE is already present)"
                fi
        else
                SUBMODULE_UPDATED=`git submodule update --init ext-deps/$module`
                if [ -z "$SUBMODULE_UPDATED" ]; then
                        echo "not needed"
                        return 0
                fi
                return 1
        fi
        return 0
}

