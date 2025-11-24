#!/bin/sh -eu

mkdir aitest-context # empty build context
./UltraGrid-"$VERSION"-x86_64.AppImage --appimage-extract

for n in archlinux ubuntu; do
        docker build -f .github/scripts/Linux/utils/Dockerfile.$n\
         -t aitest-$n aitest-context
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n\
         /AppImage/AppRun -v
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n\
         /AppImage/AppRun --tool uv-qt -h
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n\
         sh -c 'xvfb-run /AppImage/AppRun --tool uv-qt & { sleep 10; kill $!; }'
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n\
         /AppImage/AppRun --list-modules
        docker run --rm -v "$PWD"/squashfs-root:/AppImage aitest-$n\
         /AppImage/AppRun --capabilities
done
