#!/bin/sh -eu

echo "Starting Docker AppImage tests..." >&2

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

set +e
rc=0
for n in $test_list; do
        name=$(printf "%s" "$n" | tr -c '[:alnum:]' '[_*]')
        if eval wait "\$${name}_pid"; then
                echo "Docker AppImage test $name succeeded" >&2
        else
                rc=$?
                echo "Docker AppImage test $name FAILED with error code $rc" >&2
        fi
done
if [ $rc -ne 0 ]; then
        exit $rc
fi
echo "All Docker AppImage tests succeeded" >&2
