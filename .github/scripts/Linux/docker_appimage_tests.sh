#!/bin/sh -eu

mkdir aitest-context # empty build context
./UltraGrid-"$VERSION"-x86_64.AppImage --appimage-extract

run_docker_test() {
        image_name=$(echo "$1" | cut -d: -f 1)
        image_version=$(echo "$1" | cut -d: -f 2)
        dockerfile=$(mktemp)
        cat .github/scripts/Linux/utils/Dockerfile."$image_name" |
                sed "/FROM /s/:\$/$image_version/" > "$dockerfile"
        docker build -f "$dockerfile" -t aitest-"$1" aitest-context
        docker run --rm -v "$PWD"/squashfs-root/:/AppImage aitest-"$1" sh -ce '
/AppImage/AppRun -v
/AppImage/AppRun --tool uv-qt -h
xvfb-run /AppImage/AppRun --tool uv-qt & { sleep 10; kill $!; }
/AppImage/AppRun --list-modules
/AppImage/AppRun --capabilities
'
}

test_list="archlinux:latest ubuntu:22.04 ubuntu:latest"

# run the Docker tests in parallel to lower the time
for n in $test_list; do
        run_docker_test "$n" &
        name=$(printf "%s" "$n" | tr -c '[:alnum:]' '[_*]') # replace :. with _ for valid identifer
        eval "${name}_pid"=$!
done
for n in $test_list; do
        name=$(printf "%s" "$n" | tr -c '[:alnum:]' '[_*]')
        eval wait "\$${name}_pid"
done
