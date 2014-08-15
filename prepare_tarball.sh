#!/bin/sh

./autogen.sh

rm -rf .git
find -name .gitignore -print0 |xargs -0 rm

