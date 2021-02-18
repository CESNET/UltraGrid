## Install speex and speexdsp
##
## Normally try to use git submodules present in the repository. If not possible
## it tries to either clone (.git directory stripped) or to download release
## (if git not found).

for module in speex speexdsp; do
        SUBMODULE_UPDATED= # assume that submodule have not been updated
        printf "Downloading ${module}... "
        if ! command -v git >/dev/null; then
                if [ ! -d ext-deps/$module ]; then
                        echo "git not found - trying to download a release"
                        mkdir -p ext-deps
                        curl -L http://downloads.us.xiph.org/releases/speex/${module}-1.2.0.tar.gz | tar -C ext-deps -xz
                        mv ext-deps/speex*-1.2.0 ext-deps/$module
                else
                        echo "skipped (ext-deps/$module is already present)"
                fi
        elif ! git branch >/dev/null 2>&1; then
                # we are not in UltraGrid git repository but can use git to download
                if [ ! -d ext-deps/$module ]; then
                        echo "git submodule found - trying to clone"
                        git clone https://gitlab.xiph.org/xiph/$module ext-deps/$module
                else
                        echo "skipped (ext-deps/$module is already present)"
                fi
        else
                SUBMODULSE_UPDATED=`git submodule update --init ext-deps/$module`
                if [ -z "$SUBMODULE_UPDATED" ]; then
                        echo "not needed"
                fi
        fi

        printf "Configuring ${module}... "
        if [ -f ext-deps/$module/include/speex/${module}_config_types.h -a -z "$SUBMODULE_UPDATED" ]; then
                echo "not needed"
        else
                cd ext-deps/$module
                ./autogen.sh
                ./configure
                cd ../..
        fi
done

