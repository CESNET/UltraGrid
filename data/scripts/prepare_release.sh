#!/bin/sh -eu

## Prepares UG for release, does following tasks:
## - replaces verson string in configure.ac
## - creates release commit
## - from that creates release branch that contains needed scripts and removes submodules
## - pushes both
##
## usage:
## $0 <version>

VERSION=${1?call with version number}
TAGNAME=v$VERSION
BRANCH=release/$VERSION

printf "Did you update a splashscreen image? [Ny] "
read -r confirm
if [ "${confirm-N}" != y ]; then
        return 1
fi

#git submodule update --init

sed  "s/\(AC_INIT(\[UltraGrid\], \[\).*\(\], \[ultragrid-dev@cesnet.cz\]\)/\1$VERSION\2/" configure.ac
git add configure.ac
git commit -S -m "UltraGrid $VERSION"

echo "Created release commit in master branch - not pushing yet, do it manually after a review that everything is ok."

# next, we create a release branch
git branch "$BRANCH"

# create config files
./autogen.sh && rm Makefile
git add -f configure config.guess config.sub install-sh missing src/config.h.in

# remove submodules
git rm gpujpeg cineform-sdk

#rm -rf .git .gitmodules
#find -name .gitignore -print0 |xargs -0 rm
#find -name .git -print0 |xargs -0 rm -rf

git commit -S -m "Added configuration scripts"

git push origin "$BRANCH"

git tag -s "$TAGNAME"
git push upstream "refs/tags/$TAGNAME"

echo "Release branch and tag created - setup build scripts to build binary assets and release it on GitHub."

