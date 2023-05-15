#!/bin/sh -eu

readonly appname=uv-qt.app
bundle_path=$(dirname "$0")
bundle_path=$(cd "$bundle_path"/../..; pwd)
readonly bundle_path

if ! expr "x$0" : x/ >/dev/null; then
        echo "Cannot update when not running with an absolute path!" 1>&2
        exit 1
fi

if expr "x$0" : x/Volumes > /dev/null; then
        echo "Refusing update of mounted image!" 1>&2
        exit 1
fi

bundle_name=$(basename "$bundle_path")
if [ "$bundle_name" != $appname ]; then
        echo "Application name is not ending with $appname" 1>&2
        exit 1
fi

updater_dir=$(mktemp -d /tmp/ug-updater.XXXXXXXX)
cd "$updater_dir"

echo "Downloading current version"
curl -LO https://github.com/CESNET/UltraGrid/releases/download/continuous/UltraGrid-continuous.dmg

mkdir mnt
hdiutil mount -mountpoint mnt UltraGrid-continuous.dmg

rm -rf "$bundle_path"
echo "Removed old version, copying new files..."
cp -a "mnt/$appname" "$bundle_path"

umount mnt

printf "Returning to "
cd -
echo "Removing temporary data"
rm -rf "$updater_dir"

