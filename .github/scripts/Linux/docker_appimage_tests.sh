#!/bin/sh -eu

mkdir aitest-context # empty build context
./UltraGrid-"$VERSION"-x86_64.AppImage --appimage-extract

for n in archlinux ubuntu; do
        docker build -f .github/scripts/Linux/utils/Dockerfile.$n\
         -t aitest-$n aitest-context
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n sh -ce '
/AppImage/AppRun -v
/AppImage/AppRun --tool uv-qt -h
xvfb-run /AppImage/AppRun --tool uv-qt & { sleep 10; kill $!; }
/AppImage/AppRun --list-modules
/AppImage/AppRun --capabilities
'
done
