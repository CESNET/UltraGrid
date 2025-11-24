#!/bin/sh -eu

mkdir aitest-context # empty build context
./UltraGrid-"$VERSION"-x86_64.AppImage --appimage-extract

for n in archlinux:latest ubuntu:22.04 ubuntu:latest; do
        image_name=$(echo $n | cut -d: -f 1)
        image_version=$(echo $n | cut -d: -f 2)
        dockerfile=$(mktemp)
        cat .github/scripts/Linux/utils/Dockerfile."$image_name" |
                sed "/FROM /s/:\$/$image_version/" > "$dockerfile"
        docker build -f "$dockerfile" -t aitest-$n aitest-context
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-$n sh -ce '
/AppImage/AppRun -v
/AppImage/AppRun --tool uv-qt -h
xvfb-run /AppImage/AppRun --tool uv-qt & { sleep 10; kill $!; }
/AppImage/AppRun --list-modules
/AppImage/AppRun --capabilities
'
done
