#!/bin/sh -eu

mkdir aitest-context # empty build context
docker build -f .github/scripts/Linux/utils/Dockerfile.ubuntu -t aitest-ubuntu aitest-context
docker build -f .github/scripts/Linux/utils/Dockerfile.arch -t aitest-arch aitest-context
./UltraGrid-$VERSION-x86_64.AppImage --appimage-extract
docker run --rm -v $PWD/squashfs-root/:/AppImage aitest-ubuntu /AppImage/AppRun -v
docker run --rm -v $PWD/squashfs-root/:/AppImage aitest-ubuntu /AppImage/AppRun --tool uv-qt -h
docker run --rm -v $PWD/squashfs-root/:/AppImage aitest-ubuntu sh -c 'xvfb-run /AppImage/AppRun --tool uv-qt & { sleep 10; kill $!; }'
docker run --rm -v $PWD/squashfs-root/:/AppImage aitest-ubuntu /AppImage/AppRun --list-modules
docker run --rm -v $PWD/squashfs-root:/AppImage aitest-arch /AppImage/AppRun --capabilities
